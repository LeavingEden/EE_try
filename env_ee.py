# -*- coding: utf-8 -*-
"""
文件名：env_energy_efficiency.py
功能：定义DRL环境，封装RU/PN资源分配逻辑
"""

import numpy as np
import math
import random
# 导入你原本定义的数据结构和计算函数
from init_data import RU_array, PN_array


class EnergyEfficiencyEnv:
    # 参数设计
    def __init__(self, u_max=80, user_min=20, user_max=None, t_min=50):
        # 环境参数
        self.n_ru = len(RU_array) - 1  # 排除id=0的默认值
        self.n_pn = len(PN_array) - 1  # 排除id=0的默认值
        self.max_c_max = 20  # 根据init_data中PN_array的最大C_max设定

        # 动作空间：RU开关 (0/1) + PN的C_ai分配 (0 ~ C_max)
        # 总动作维度：RU数量 + PN数量
        self.action_dim = self.n_ru + self.n_pn * 2

        # -----------------------------
        # 观测/状态（用于 Transformer）
        # -----------------------------
        # token 特征维度：
        # RU token:  [sta, ru_x_norm, ru_y_norm] -> ru_dim=3
        # PN token:  [sta, pn_x_norm, pn_y_norm, C_ai_norm, C_bbu_norm, C_max_norm] -> pn_dim=6
        # User token:[user_x_norm, user_y_norm, active] -> user_dim=3
        self.ru_dim = 3
        self.pn_dim = 6
        self.user_dim = 3
        self.x_max = 1400.0
        self.y_max = 1200.0
        self.U_max = int(u_max)          # 允许的最大用户数（padding到固定长度）
        self.user_min = int(user_min)    # 每步至少激活多少用户
        self.user_max = int(user_max) if user_max is not None else self.U_max
        self.user_min = max(1, min(self.user_min, self.U_max))
        self.user_max = max(self.user_min, min(self.user_max, self.U_max))
        self.max_steps = 5       # 每个 episode 的时间步数（多步时序）
        self.user_step_delta = 3  # 每步用户数量最大变化幅度（越小越平缓）

        # 状态向量是固定长度的 flatten tokens，方便主循环用 A2C 收集 batch
        # state_dim = M*ru_dim + N*pn_dim + U_max*user_dim
        self.state_dim = self.n_ru * self.ru_dim + self.n_pn * self.pn_dim + self.U_max * self.user_dim

        # 用于保存原始数据（基准）
        self.original_ru = RU_array[1:]
        self.original_pn = PN_array[1:]

        # 固定参数设计
        self.ON = 1
        self.OFF = 0
        self.B_J0 = 2.5e5
        self.B_J = 1e5
        self.P_k = 0.1
        self.Alpha = 3
        self.H_k = 1e-6
        self.I_k = 1e-14
        self.Sigma2 = 4e-15
        self.tau = 20
        self.D = 2000
        self.t_min = float(t_min)

        self.t = 0
        self.user_array = None
        self.current_active_users = self.user_min

    def _make_user_slot(self, uid, active, pos):
        # active=0 表示 padding/未激活用户；在 compute_ee/连接逻辑里会被跳过
        return {
            "id": uid,
            "active": int(active),
            "pos": pos if active == 1 else (0.0, 0.0),
            "dis": None,
            "Tr": 0.0,
            "Tp": 0.0,
            "Tw": 0.0,
            "Bk": 0.0,
            "Rk": None,
            "Pn": -1,
        }

    def _reset_users(self):
        # 初始化用户集合：数量随 reset / step 变化
        K = random.randint(self.user_min, self.user_max)
        self.current_active_users = K
        users = [{"id": 0, "active": 0, "pos": (0.0, 0.0)}]
        for uid in range(1, self.U_max + 1):
            active = 1 if uid <= K else 0
            if active == 1:
                x = float(random.randint(0, int(self.x_max)))
                y = float(random.randint(0, int(self.y_max)))
                users.append(self._make_user_slot(uid, 1, (x, y)))
            else:
                users.append(self._make_user_slot(uid, 0, (0.0, 0.0)))
        return users

    def _update_users(self):
        # 更平缓的随机游走 + 用户数量缓慢变化（降低非平稳性）
        delta = random.randint(-self.user_step_delta, self.user_step_delta)
        K = int(np.clip(self.current_active_users + delta, self.user_min, self.user_max))
        self.current_active_users = K
        for uid in range(1, self.U_max + 1):
            user = self.user_array[uid]
            if uid <= K:
                # active user：小幅位置扰动
                x, y = user["pos"]
                if user.get("active", 0) != 1:
                    x = float(random.randint(0, int(self.x_max)))
                    y = float(random.randint(0, int(self.y_max)))
                else:
                    x = float(np.clip(x + random.uniform(-20, 20), 0.0, self.x_max))
                    y = float(np.clip(y + random.uniform(-20, 20), 0.0, self.y_max))
                user.update({
                    "active": 1,
                    "pos": (x, y),
                    "dis": None,
                    "Pn": -1,
                    "Bk": 0.0,
                    "Rk": None,
                    "Tr": 0.0,
                    "Tp": 0.0,
                    "Tw": 0.0,
                })
            else:
                # inactive user：保持 padding 状态
                user.update({
                    "active": 0,
                    "pos": (0.0, 0.0),
                    "dis": None,
                    "Pn": -1,
                    "Bk": 0.0,
                    "Rk": None,
                    "Tr": 0.0,
                    "Tp": 0.0,
                    "Tw": 0.0,
                })

    def _build_observation(self, ru_array, pn_array):
        # RU tokens: 每个 token = [sta, ru_x_norm, ru_y_norm] (ru_dim=3)
        ru_tokens = []
        for ru in ru_array[1:]:
            x, y = ru["pos"]
            ru_tokens.append([
                float(ru["sta"]),
                float(x) / self.x_max,
                float(y) / self.y_max,
            ])

        # PN tokens: 每个 token = [sta, pn_x_norm, pn_y_norm, C_ai_norm, C_bbu_norm, C_max_norm] (pn_dim=6)
        pn_tokens = []
        for pn in pn_array[1:]:
            x, y = pn["pos"]
            c_max = pn["C_max"] if "C_max" in pn else self.max_c_max
            c_max = float(c_max) if c_max > 0 else 1.0
            c_ai_norm = float(pn["C_ai"]) / c_max
            c_bbu_norm = float(pn["C_bbu"]) / c_max
            pn_tokens.append([
                float(pn["sta"]),
                float(x) / self.x_max,
                float(y) / self.y_max,
                c_ai_norm,
                c_bbu_norm,
                float(c_max) / float(self.max_c_max),
            ])

        # User tokens（固定 U_max，inactive 做 padding）
        user_tokens = []
        for uid in range(1, self.U_max + 1):
            u = self.user_array[uid]
            x, y = u["pos"]
            user_tokens.append([
                float(x) / self.x_max,
                float(y) / self.y_max,
                float(u.get("active", 0)),
            ])

        # flatten：不要直接 np.array(ru_tokens + pn_tokens + user_tokens)，
        # 因为 ru/pn/user 的 token 维度不同（3 vs 6），会导致不规则嵌套序列错误。
        features = []
        for t in ru_tokens:
            features.extend(t)
        for t in pn_tokens:
            features.extend(t)
        for t in user_tokens:
            features.extend(t)

        state = np.asarray(features, dtype=np.float32)
        # 保险：确保长度匹配 state_dim
        if state.size != self.state_dim:
            raise ValueError(f"observation dim mismatch: got {state.size}, expected {self.state_dim}")
        return state
    def reset(self):
        """
        重置环境到初始状态
        返回状态向量
        """
        self.t = 0
        self.user_array = self._reset_users()

        # 初始配置：所有 RU/P N 使用基准值
        ru_array = [{"id": 0, "sta": 0, "pos": (0, 0), "Km": [], "Pm": None}]
        for ru in self.original_ru:
            ru_array.append({
                "id": ru["id"],
                "sta": 1,
                "pos": ru["pos"],
                "Km": [],
                "Pm": None,
            })

        pn_array = [{"id": 0, "sta": 1, "pos": (0, 0), "Jm": [], "Kj": [], "C_ai": 20, "C_bbu": 20, "C_max": 60, "Pj": 0}]
        for pn in self.original_pn:
            c_ai = pn["C_ai"]
            c_bbu = pn["C_bbu"]
            sta = 1 if (c_ai > 0 and c_bbu > 0) else 0
            pn_array.append({
                "id": pn["id"],
                "sta": sta,
                "pos": pn["pos"],
                "Jm": [],
                "Kj": [],
                "C_ai": c_ai,
                "C_bbu": c_bbu,
                "C_max": pn["C_max"],
                "Pj": None,
            })

        return self._build_observation(ru_array, pn_array)

    def step(self, action_vector):
        """
        执行动作
        action_vector: shape (n_ru + n_pn,)
                     前n_ru个元素是RU的开关概率(需二值化)
                     后n_pn个元素是PN分配给AI的比例 [0, 1]
        """
        self.t += 1

        # 1. 解析动作向量
        ru_actions = action_vector[:self.n_ru]  # [0, 1] 概率
        pn_ai_ratio = action_vector[self.n_ru: self.n_ru + self.n_pn]  # C_ai比例 [0,1]
        pn_bbu_ratio = action_vector[self.n_ru + self.n_pn:]  # C_bbu比例 [0,1]

        # 2. 构建新的 RU_array 和 PN_array
        new_ru_array = [{"id": 0, "sta": 0, "pos": (0, 0), "Km": [], "Pm": None}]
        new_pn_array = [
            {"id": 0, "sta": 1, "pos": (0, 0), "Jm": [], "Kj": [], "C_ai": 20, "C_bbu": 20, "C_max": 60, "Pj": 0}]

        # 处理 RU (动作 > 0.5 视为开启)
        for i, act in enumerate(ru_actions):
            ru_id = i + 1
            sta = 1 if act > 0.5 else 0
            # 保留原始位置信息
            orig_ru = next(r for r in RU_array if r["id"] == ru_id)
            new_ru = {
                "id": ru_id,
                "sta": sta,
                "pos": orig_ru["pos"],
                "Km": [],  # 会在计算流程中重新分配
                "Pm": None
            }
            new_ru_array.append(new_ru)

        for i in range(self.n_pn):
            pn_id = i + 1
            orig_pn = next(p for p in PN_array if p["id"] == pn_id)
            c_max = orig_pn["C_max"]

            # 步骤1：根据各自比例计算原始C_ai/C_bbu（取整）
            # 动作映射：ratio * c_max → 原始分配值
            c_ai_raw = int(round(pn_ai_ratio[i] * c_max))
            c_bbu_raw = int(round(pn_bbu_ratio[i] * c_max))

            # 步骤2：约束规则（可根据需求调整）
            # 规则1：C_ai/C_bbu ≥ 0（允许为0，实现「单独关闭」）
            c_ai = max(0, c_ai_raw)
            c_bbu = max(0, c_bbu_raw)

            # 规则2：总和不超过C_max（若超则按比例缩减）
            total = c_ai + c_bbu
            if total > c_max:
                # 方式1：等比例缩减（保持分配比例）
                scale = c_max / total
                c_ai = int(round(c_ai * scale))
                c_bbu = int(round(c_bbu * scale))
                # 兜底：避免四舍五入后总和仍超（可选）
                if c_ai + c_bbu > c_max:
                    c_bbu = c_max - c_ai

            # 假设只要分配了资源，PN就是激活的 (或者根据c_ai/c_bbu是否为0判断)
            sta = 1 if (c_ai > 0 and c_bbu > 0) else 0

            new_pn = {
                "id": pn_id,
                "sta": sta,
                "pos": orig_pn["pos"],
                "Jm": [],  # 会在计算流程中重新计算
                "Kj": [],
                "C_ai": c_ai,
                "C_bbu": c_bbu,
                "C_max": c_max,
                "Pj": None
            }
            new_pn_array.append(new_pn)

        # 3. 调用核心计算逻辑 (复用你test.py里的逻辑)
        # 注意：这里需要把计算函数移到这个类里，或者单独作为一个utils文件
        # 为了演示，我直接调用逻辑（你需要把下面的compute_ee函数定义好）
        ee_value,sum_pm,sum_pj, is_valid = self.compute_ee(new_ru_array, new_pn_array)

        # 4. 定义奖励函数（稳定版：软惩罚替代硬惩罚，减少断崖式波动）
        reward = ee_value
        active_users = max(1, sum(1 for u in self.user_array[1:] if u.get("active", 0) == 1))
        served_users = int(round(ee_value * (sum_pm + sum_pj) / (self.tau * 30))) if (sum_pm + sum_pj) > 0 else 0
        served_ratio = float(np.clip(served_users / active_users, 0.0, 1.0))

        # PN 休眠惩罚：仅当分配出现“单边资源”时惩罚，防止大量无效动作
        one_side_alloc = 0
        for pn in new_pn_array[1:]:
            if (pn["C_ai"] > 0 and pn["C_bbu"] == 0) or (pn["C_ai"] == 0 and pn["C_bbu"] > 0):
                one_side_alloc += 1

        if not is_valid:
            reward -= 20.0
        reward -= 5.0 * (1.0 - served_ratio)
        reward -= 1.5 * one_side_alloc

            # 5. 构造新的状态向量 (供下次使用)
        next_state = []
        for ru in new_ru_array[1:]:
            next_state.append(ru['sta'])
        for pn in new_pn_array[1:]:
            next_state.append(pn['sta'])
            next_state.append(pn['C_ai'])
            next_state.append(pn['C_bbu'])

        next_state = np.array(next_state, dtype=np.float32)

        done = self.t >= self.max_steps

        # 用户集合在“动作后”更新，形成时序变化
        self._update_users()

        # 构造下一状态（与 Transformer 观测一致）
        next_state = self._build_observation(new_ru_array, new_pn_array)

        temp_user = [dict(u) for u in self.user_array]
        # --- 新增代码：构建 Info 字典 ---
        # 在 return 语句之前添加以下代码
        info = {
            'power_ru': sum_pm,  # sum_pm 是在 compute_ee 中计算出的 RU 总功耗
            'power_pn': sum_pj,  # sum_pj 是在 compute_ee 中计算出的 PN 总功耗
            'user_array': temp_user,
            'users_served': served_users,
            'served_ratio': served_ratio,
            'one_side_alloc': one_side_alloc,
        }
        return next_state, reward, done, info, {}

    def compute_ee(self, ru_array, pn_array):
        """
        这个函数是把你test.py里计算EE的核心逻辑搬过来。
        注意：需要处理数据引用问题，避免污染全局变量。
        """
        try:
            # 1. 做连接关系 (RU分配给PN, User分配给RU)
            # 注意：这里需要传入副本，或者确保函数内部不修改原始引用
            temp_pn = [dict(pn) for pn in pn_array]  # 浅拷贝
            temp_ru = [dict(ru) for ru in ru_array]
            temp_user = [dict(u) for u in self.user_array]  # 用户数据通常会在 step/reset 内更新

            # 执行连接逻辑 (复制你test.py里的函数逻辑)
            temp_pn = self.RU_to_PN(temp_ru, temp_pn)
            temp_user, temp_ru = self.User_to_RU(temp_user, temp_ru)
            temp_user, temp_pn = self.User_to_PN(temp_user, temp_ru, temp_pn)

            # 2. 计算时延和功率
            # 计算用户时延
            for u in temp_user:
                if u["id"] == 0:
                    continue
                # padding/未激活用户：不参与连接与服务判定
                if u.get("active", 1) != 1:
                    u["Tr"], u["Tp"], u["Tw"] = 1000, 1000, 1000
                    continue
                # 检查用户是否关联了PN，否则跳过（Pn=-1 表示未连接）
                if ("Pn" not in u) or (u["Pn"] is None) or (u["Pn"] == -1):
                    u["Tr"], u["Tp"], u["Tw"] = 1000, 1000, 1000
                    continue
                u_rk, u_tr = self.T_tr(u, temp_pn)
                u_tp = self.T_process(u, temp_pn)
                u_tw = self.T_wired(u)
                u["Rk"] = u_rk
                u["Tr"] = u_tr
                u["Tp"] = u_tp
                u["Tw"] = u_tw

            # 计算RU功率
            for r in temp_ru:
                if r["id"] == 0:
                    continue
                r_pm = self.P_m(r)
                r["Pm"] = r_pm

            # 计算PN功率
            for p in temp_pn:
                p_pj = self.P_j(p, temp_user)
                p["Pj"] = p_pj
            sum_pm = 0
            sum_pj = 0
            # 3. 计算EE
            ee, sum_pm, sum_pj = self.EE(
                temp_user,
                temp_ru,
                temp_pn,
                tau=self.tau,
                fps=30,
                t_min=self.t_min,
            )
            # 简单的合法性检查：EE不能为负或无穷
            if ee <= 0 or math.isinf(ee) or math.isnan(ee):
                return 0.0, 0.0, 0.0, False

            return ee, sum_pm, sum_pj, True

        except Exception as e:
            print(f"计算错误: {e}")
            return 0, 0, 0, False

    # --- 下面是把你test.py里的函数复制粘贴到这里 ---
    # (为了保持代码完整性，这里只列出声明，实际使用需把函数体复制过来)
    def Get_pos(self, cen_x, cen_y, length):
        x_list = [0, 0, 0, 0, 0, 0]
        y_list = [0, 0, 0, 0, 0, 0]
        x_list[0] = cen_x + length
        x_list[1] = cen_x + length / 2
        x_list[2] = cen_x - length / 2
        x_list[3] = cen_x - length
        x_list[4] = x_list[2]
        x_list[5] = x_list[1]
        y_list[0] = cen_y
        y_list[1] = cen_y + length / 2 * 1.73
        y_list[2] = y_list[1]
        y_list[3] = y_list[0]
        y_list[4] = cen_y - length / 2 * 1.73
        y_list[5] = y_list[4]
        pos = [(x, y) for x, y in zip(x_list, y_list)]
        return pos

    def Random_pos(self, x_min, x_max, y_min, y_max, num_points, max_points = 500):
        # 1. 参数校验
        if num_points < 0:
            raise ValueError("点数量不能为负数")
        if num_points > max_points:
            raise ValueError(f"点数量{num_points}超过最大限制{max_points}")

        # 2. 计算矩形范围（处理闭区间，randint左闭右开→x_max+1包含x_max）
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            raise ValueError("矩形范围无效（x_min需<x_max，y_min需<y_max）")

        # 3. 生成固定数量的整数坐标（均匀分布，保证空间随机性）
        # 注：randint(low, high)→low≤x<high，故x_max+1可包含x_max
        x_cords = np.random.randint(low=x_min, high=x_max + 1, size=num_points).tolist()
        y_cords = np.random.randint(low=y_min, high=y_max + 1, size=num_points).tolist()

        # 4. 组合为坐标对
        points = list(zip(x_cords, y_cords))
        print(x_cords, '\n')
        print(points, '\n')
        return points
    def RU_to_PN(self, ru_array, pn_array):
        # 1. 清空所有PN的Jm数组（避免历史数据）
        for pn in pn_array:
            pn["Jm"].clear()  # 清空数组

        # 2. 遍历每个RU，计算与所有PN的距离，找到最近的PN
        for ru in ru_array:
            # 获取RU的编号和位置
            ru_id = ru["id"]
            ru_x, ru_y = ru["pos"]
            # RU休眠直接跳过
            if ru.get("sta", self.OFF) != 1:
                # print("tiaoguo", '\n')
                continue  # 跳过休眠RU
            # 初始化最小距离和最近PN
            min_distance = float("inf")  # 初始为无穷大
            nearest_pn = None

            # 遍历所有PN，计算距离
            for pn in pn_array:
                pn_id = pn["id"]
                pn_x, pn_y = pn["pos"]

                # 计算欧氏距离（通信场景常用，若需曼哈顿距离可替换为abs(ru_x-pn_x)+abs(ru_y-pn_y)）
                distance = math.hypot(ru_x - pn_x, ru_y - pn_y)  # 等价于sqrt((x1-x2)²+(y1-y2)²)

                # 更新最小距离和最近PN（严格小于才更新，确保唯一最近PN；若需处理相等距离见扩展）
                if distance < min_distance:
                    min_distance = distance
                    nearest_pn = pn

            # 3. 将RU编号加入最近PN的Jm数组
            if nearest_pn is not None:
                nearest_pn["Jm"].append(ru_id)  # 数组允许重复，按遍历顺序添加

        return pn_array

    def User_to_RU(self, user_array, ru_array):
        # 1. 清空所有RU的Km数组（避免历史数据）
        for ru in ru_array:
            ru["Km"].clear()  # 清空数组

        # 2. 初始化所有User的dis为None（避免历史数据）
        for user in user_array:
            user["dis"] = None  # 重置距离

        # 3. 遍历每个User，计算与所有激活RU的距离，找到最近的激活RU
        for user in user_array:
            # 获取User的编号和位置（跳过User_array[0]的默认值，id=0）
            if user["id"] == 0:
                continue  # 跳过默认值
            if user.get("active", 1) != 1:
                continue  # padding/未激活用户不参与连接

            user_id = user["id"]
            user_x, user_y = user["pos"]

            # 初始化最小距离和最近激活RU
            min_distance = float("inf")  # 初始为无穷大
            nearest_active_ru = None

            # 遍历所有RU，仅处理激活的RU（status=1）
            for ru in ru_array:
                # 仅考虑激活的RU（status=1）
                if ru.get("sta", self.OFF) != 1:
                    # print("tiaoguo",'\n')
                    continue  # 跳过休眠RU

                # 获取激活RU的编号和位置
                ru_id = ru["id"]
                ru_x, ru_y = ru["pos"]

                # 计算User到激活RU的欧氏距离
                distance = math.hypot(user_x - ru_x, user_y - ru_y)

                # 更新最小距离和最近激活RU
                if distance < min_distance:
                    min_distance = distance
                    nearest_active_ru = ru

            # 4. 将User编号加入最近激活RU的Km数组，并记录距离到User的dis字段
            if nearest_active_ru is not None:
                # 加入RU的Km数组
                nearest_active_ru["Km"].append(user_id)
                # 记录距离到User的dis字段
                user["dis"] = round(min_distance, 1)  # 存储最短距离

        return user_array, ru_array

    def User_to_PN(self, user_array, ru_array, pn_array):
        # 1. 清空所有PN的Kj数组（避免历史数据）
        for pn in pn_array:
            pn["Kj"].clear()

        # 2. 将RU的用户数组放入PN中
        for pn in pn_array:
            for ru in pn["Jm"]:
                if pn.get("sta", self.OFF) != 1:
                    pn0 = next((p for p in pn_array if p["id"] == 0), None)
                    pn0["Kj"] = list(set(pn0["Kj"]).union(ru_array[ru]["Km"]))
                else:
                    pn["Kj"] = list(set(pn["Kj"]).union(ru_array[ru]["Km"]))
        for pn in pn_array:
            if pn["Kj"] != [] and pn["id"] != 0:
                bk = self.B_J / len(pn["Kj"])
                for uid in pn["Kj"]:
                    user = next((user for user in user_array if user["id"] == uid), None)
                    if user is None or user.get("active", 1) != 1:
                        continue
                    user["Bk"] = round(bk, 1)
                    user["Pn"] = pn["id"]
            if pn["Kj"] != [] and pn["id"] == 0:
                bk0 = self.B_J0 / len(pn["Kj"])
                for uid in pn["Kj"]:
                    user = next((user for user in user_array if user["id"] == uid), None)
                    if user is None or user.get("active", 1) != 1:
                        continue
                    user["Bk"] = round(bk0, 1)
                    user["Pn"] = pn["id"]
        return user_array, pn_array

    def T_tr(self, user, pn_array, d=1000, pk=0.1, alpha=3, hk=1e-6, ik=1e-14, sigma2=1e-15):
        # 利用公式（1）
        bi = 0
        c_bbu = 0
        for pn in pn_array:
            if user["id"] in pn["Kj"]:
                bi = len(pn["Kj"])
                c_bbu = pn["C_bbu"]
                break
            else:
                bi = 0
        # 防御：如果未找到关联且导致 c_bbu=0，则该用户视为不可服务（由上游逻辑控制）
        if c_bbu == 0:
            eff = 0.0
        else:
            eff = min(1, bi / c_bbu)
        rk_max = 2e5
        bk = user["Bk"]
        dis = user["dis"]
        loss = dis ** (-alpha)
        received_power = pk * hk * loss
        sinr = received_power / (ik + sigma2)
        rk = round(bk * float(np.log2(1 + sinr)) * eff, 1)
        if rk != 0:
            if rk > rk_max:
                rk = rk_max
            tr = round(d / rk * 1000, 2)
            return rk, tr
        else:
            tr = 1000
            return rk, tr

    def T_process(self, user, pn_array):
        bj = 0
        c_ai = 0
        for pn in pn_array:
            if user["id"] in pn["Kj"]:
                bj = len(pn["Kj"])
                c_ai = pn["C_ai"]
                break
            else:
                bj = 0
        alpha_tp = 0.3051
        l0 = 1.0524
        if c_ai != 0:
            tp = round(float(alpha_tp * bj / c_ai) + l0, 2)
        else:
            tp = 1000
        return tp

    def T_wired(self, user):
        if user["Pn"] == 0:
            return 30
        else:
            return 0

    def P_m(self, ru, pk=0.1):
        eta_pa = 0.45
        p_fix = 0.5
        p_sleep = 0.2
        user_num = len(ru["Km"])
        if user_num == 0:
            pm = p_sleep
        else:
            pm = p_fix + user_num * pk / eta_pa
        return round(pm, 2)

    def P_j(self, pn, user_array, d=1000):
        # BBU部分
        esp_bbu = 3e-13
        sum_rk = 0
        FBBU = 10
        gam = 0.8
        for uid in pn["Kj"]:
            user = next((u for u in user_array if u["id"] == uid), None)
            if user and user["Rk"] is None:
                sum_rk += user["Rk"]

        if pn["C_bbu"] != 0:
            f_bbu = sum_rk * FBBU / pn["C_bbu"] / gam
        else:
            f_bbu = 0
        p_bbu = esp_bbu * f_bbu * f_bbu
        # AI部分
        esp_ai = 3e-13
        sum_u = len(pn["Kj"])
        FPS = 30
        FAI = 10
        if pn["C_ai"] != 0:
            f_ai = sum_u * FAI * d * FPS / pn["C_ai"] / gam
        else:
            f_ai = 0
        p_ai = esp_ai * f_ai * f_ai
        # 总功耗
        pj_bbu_sta = 0.8
        pj_ai_sta = 0.8
        pj_sta = 20
        pj_sleep = 10
        pj_active = pn["C_ai"] * (p_ai + pj_ai_sta) + pn["C_bbu"] * (p_bbu + pj_bbu_sta) + pj_sta
        if pn["sta"] != 0:
            return round(pj_active, 2)
        else:
            return pj_sleep

    def EE(self, user_array, ru_array, pn_array, tau=20, fps=30, t_min=50):
        sum_user = 0
        for user in user_array:
            if user["id"] == 0:
                continue
            if user.get("active", 1) != 1:
                continue
            sum_t = user["Tr"] + user["Tp"] + user["Tw"]
            if sum_t < t_min:
                sum_user += 1
        sum_pm = 0
        for ru in ru_array:
            if ru["id"] == 0:
                continue
            sum_pm += ru["Pm"]
        sum_pj = 0
        for pn in pn_array:
            sum_pj += pn["Pj"]
        ee = 0
        if sum_pj + sum_pm > 0:
            ee = tau * fps * sum_user / (sum_pm + sum_pj)
        return ee, sum_pm, sum_pj
