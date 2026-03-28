"""
传统负载阈值休眠基线算法（可与 DRL 结果对比）

核心逻辑：
1) 通过双阈值（th_act / th_deact）控制 PN 休眠与唤醒（带迟滞）。
2) 根据当前用户负载进行简单负载均衡，生成 RU 开关与 PN 资源分配动作。
3) 在与 DRL 相同的动态用户环境中运行，输出 EE 曲线并绘制对比图。
"""

import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from env_ee import EnergyEfficiencyEnv
from init_data import PN_array, RU_array


class ThresholdSleepController:
    """传统阈值休眠控制器（hysteresis + 负载驱动资源分配）。"""

    def __init__(
        self,
        env: EnergyEfficiencyEnv,
        th_act: float = 0.65,
        th_deact: float = 0.25,
    ):
        self.env = env
        self.th_act = th_act
        self.th_deact = th_deact
        self.n_ru = env.n_ru
        self.n_pn = env.n_pn

        # 初始时默认 PN 全部激活（与系统初始配置一致）
        self.pn_state = {pid: 1 for pid in range(1, self.n_pn + 1)}
        self.ru_to_pn = self._build_ru_to_pn_map()
        self.pn_neighbors = self._build_pn_neighbors(k=2)

    def _build_ru_to_pn_map(self) -> Dict[int, int]:
        """按几何最近原则建立 RU->PN 固定映射。"""
        mapping = {}
        pns = PN_array[1:]
        for ru in RU_array[1:]:
            ru_id = ru["id"]
            rx, ry = ru["pos"]
            nearest_pn = min(
                pns,
                key=lambda pn: math.hypot(rx - pn["pos"][0], ry - pn["pos"][1]),
            )
            mapping[ru_id] = nearest_pn["id"]
        return mapping

    def _build_pn_neighbors(self, k: int = 2) -> Dict[int, List[int]]:
        """给每个 PN 选取 k 个最近邻 PN。"""
        neighbors = {}
        pns = PN_array[1:]
        for pn in pns:
            pid = pn["id"]
            px, py = pn["pos"]
            cand = []
            for other in pns:
                if other["id"] == pid:
                    continue
                ox, oy = other["pos"]
                cand.append((math.hypot(px - ox, py - oy), other["id"]))
            cand.sort(key=lambda x: x[0])
            neighbors[pid] = [nid for _, nid in cand[:k]]
        return neighbors

    def _assign_users_to_ru(self, users: List[dict]) -> Dict[int, int]:
        """把 active 用户按最近 RU 分配，返回每个 RU 的用户数。"""
        ru_load = {rid: 0 for rid in range(1, self.n_ru + 1)}
        ru_pos = {ru["id"]: ru["pos"] for ru in RU_array[1:]}
        for u in users[1:]:
            if u.get("active", 0) != 1:
                continue
            ux, uy = u["pos"]
            nearest_ru = min(
                ru_pos.keys(),
                key=lambda rid: math.hypot(
                    ux - ru_pos[rid][0], uy - ru_pos[rid][1]
                ),
            )
            ru_load[nearest_ru] += 1
        return ru_load

    def build_action(self, users: List[dict]) -> np.ndarray:
        """
        构造动作向量：
        [ru_on/off(0/1), pn_ai_ratio(0~1), pn_bbu_ratio(0~1)]
        """
        ru_load = self._assign_users_to_ru(users)
        active_users = max(
            1, sum(1 for u in users[1:] if u.get("active", 0) == 1)
        )

        # RU 规则：有用户则开，无用户则关（传统节能策略）
        ru_actions = np.zeros(self.n_ru, dtype=np.float32)
        for rid in range(1, self.n_ru + 1):
            ru_actions[rid - 1] = 1.0 if ru_load[rid] > 0 else 0.0

        # 聚合到 PN 负载
        pn_users = {pid: 0 for pid in range(1, self.n_pn + 1)}
        for rid, ucnt in ru_load.items():
            pn_users[self.ru_to_pn[rid]] += ucnt

        # 归一化负载比例 li，作为阈值判断依据
        # 传统方案常用“占容量比例”；这里用动态平均容量近似（可用于异构规模）
        avg_capacity = max(1.0, active_users / float(self.n_pn))
        pn_load_ratio = {
            pid: min(1.5, pn_users[pid] / avg_capacity)
            for pid in range(1, self.n_pn + 1)
        }

        # 迟滞控制：deact / act
        for pid in range(1, self.n_pn + 1):
            load = pn_load_ratio[pid]
            if self.pn_state[pid] == 1 and load < self.th_deact:
                self.pn_state[pid] = 0
            elif self.pn_state[pid] == 0:
                # 睡眠态时，若邻居高负载，则唤醒
                neigh_high = any(
                    pn_load_ratio[nid] > self.th_act
                    for nid in self.pn_neighbors[pid]
                )
                if neigh_high:
                    self.pn_state[pid] = 1

        # 安全兜底：至少保留一个 PN 激活，避免服务全失效
        if sum(self.pn_state.values()) == 0:
            best_pid = max(
                pn_load_ratio.keys(),
                key=lambda pid: pn_load_ratio[pid],
            )
            self.pn_state[best_pid] = 1

        # PN 资源分配：
        # 激活态：C_ai/C_bbu 均分并随负载线性增长（最小给 1，保证不“单边休眠”）
        # 睡眠态：都为 0，对应 p_sleep
        pn_ai_ratio = np.zeros(self.n_pn, dtype=np.float32)
        pn_bbu_ratio = np.zeros(self.n_pn, dtype=np.float32)
        for pid in range(1, self.n_pn + 1):
            c_max = PN_array[pid]["C_max"]
            if self.pn_state[pid] == 0:
                c_ai = 0
                c_bbu = 0
            else:
                li = max(0.1, min(1.0, pn_load_ratio[pid]))
                total = max(2, int(round(li * c_max)))
                c_ai = max(1, total // 2)
                c_bbu = max(1, total - c_ai)
                if c_ai + c_bbu > c_max:
                    c_bbu = c_max - c_ai
            pn_ai_ratio[pid - 1] = float(c_ai) / float(c_max)
            pn_bbu_ratio[pid - 1] = float(c_bbu) / float(c_max)

        action = np.concatenate(
            [ru_actions, pn_ai_ratio, pn_bbu_ratio]
        ).astype(np.float32)
        return action


def run_threshold_baseline(episodes: int = 2000):
    env = EnergyEfficiencyEnv()
    controller = ThresholdSleepController(env, th_act=0.65, th_deact=0.25)

    ee_history = []
    for ep in range(episodes):
        _ = env.reset()
        done = False
        ep_rewards = []
        while not done:
            action = controller.build_action(env.user_array)
            _, reward, done, _, _ = env.step(action)
            ep_rewards.append(reward)
        ee_history.append(float(np.mean(ep_rewards)))

        if (ep + 1) % 100 == 0:
            avg_100 = np.mean(ee_history[-100:])
            print(
                f"[Threshold] Episode {ep+1}/{episodes}, "
                f"Avg EE (last 100): {avg_100:.5f}"
            )

    return np.asarray(ee_history, dtype=np.float32)


def smooth_curve(arr: np.ndarray, window: int = 200):
    if arr.size < window:
        return np.arange(arr.size), arr
    kernel = np.ones(window, dtype=np.float32) / window
    ma = np.convolve(arr, kernel, mode="valid")
    x = np.arange(window - 1, arr.size)
    return x, ma


def main():
    episodes = 2000
    env_cfg = EnergyEfficiencyEnv()
    baseline = run_threshold_baseline(episodes=episodes)
    np.save("ee_history_threshold.npy", baseline)

    # 尝试读取 DRL 历史（由 train.py 导出）
    drl = None
    try:
        drl = np.load("ee_history_drl.npy")
        print("已读取 DRL 曲线: ee_history_drl.npy")
    except FileNotFoundError:
        print("未找到 ee_history_drl.npy，仅绘制阈值基线。")

    plt.figure(figsize=(11, 6))
    bx, by = smooth_curve(baseline, window=200)
    plt.plot(baseline, alpha=0.2, label="Threshold (raw)")
    plt.plot(bx, by, linewidth=2.0, label="Threshold (MA200)")

    if drl is not None and drl.size > 0:
        dx, dy = smooth_curve(drl.astype(np.float32), window=200)
        plt.plot(drl, alpha=0.2, label="Transformer-A2C (raw)")
        plt.plot(dx, dy, linewidth=2.2, label="Transformer-A2C (MA200)")
        print(f"Threshold mean(last100): {np.mean(baseline[-100:]):.5f}")
        print(f"Transformer-A2C mean(last100): {np.mean(drl[-100:]):.5f}")

    plt.title("EE Comparison: DRL vs Threshold Sleep Baseline")
    plt.xlabel("Episode")
    plt.ylabel("EE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ee_compare_drl_vs_threshold.png")
    plt.show()

    # 额外输出：柱状图对比“训练末期(last100)”
    labels = ["Threshold", "Transformer-A2C"]
    th_last100 = float(np.mean(baseline[-100:])) if baseline.size >= 100 else float(np.mean(baseline))
    drl_last100 = None
    if drl is not None and drl.size > 0:
        drl_last100 = float(np.mean(drl[-100:])) if drl.size >= 100 else float(np.mean(drl))

    values = [th_last100]
    bar_labels = ["Threshold"]
    if drl_last100 is not None:
        values.append(drl_last100)
        bar_labels.append("Transformer-A2C")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(bar_labels, values, color=["#4e79a7", "#f28e2b"][: len(values)])
    plt.ylabel("EE (last 100 episodes mean)")
    plt.title("Final-Stage EE Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # 在柱子上标注 EE 数值与用户人数信息
    user_note = f"Users: dynamic [{env_cfg.user_min}, {env_cfg.U_max}]"
    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 0.8, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        plt.text(x, max(1.0, y * 0.08), user_note, ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig("ee_compare_bar_final.png")
    plt.show()
    print("已保存柱状图: ee_compare_bar_final.png")


if __name__ == "__main__":
    main()
