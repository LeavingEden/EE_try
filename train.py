# -*- coding: utf-8 -*-
"""
文件名：train_a2c.py
功能：使用A2C算法训练能效优化模型
依赖库：pip install torch numpy matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli, Beta
from env_ee import EnergyEfficiencyEnv

# --- 1. Transformer Actor-Critic（编码 RU/PN/User token） ---
class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        n_ru,
        n_pn,
        U_max,
        ru_dim,
        pn_dim,
        user_dim,
        d_model=128,
        nhead=8,
        num_layers=2,
    ):
        super().__init__()
        self.n_ru = n_ru
        self.n_pn = n_pn
        self.U_max = U_max
        self.ru_dim = ru_dim
        self.pn_dim = pn_dim
        self.user_dim = user_dim

        self.ru_proj = nn.Linear(ru_dim, d_model)
        self.pn_proj = nn.Linear(pn_dim, d_model)
        self.user_proj = nn.Linear(user_dim, d_model)

        # 三类 token 的类型嵌入（关键：异构资源/实体交互）
        self.type_emb = nn.Parameter(torch.randn(3, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor heads
        self.ru_head = nn.Linear(d_model, 1)          # 每个 RU 一个 logit
        self.pn_ai_alpha_head = nn.Linear(d_model, 1)
        self.pn_ai_beta_head = nn.Linear(d_model, 1)
        self.pn_bbu_alpha_head = nn.Linear(d_model, 1)
        self.pn_bbu_beta_head = nn.Linear(d_model, 1)

        # Critic head（masked pooling -> value）
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.softplus = nn.Softplus()

    def _parse_state(self, state_vec):
        """
        state_vec: (B, state_dim)
        返回：
          ru_tokens:   (B, M, ru_dim)
          pn_tokens:   (B, N, pn_dim)
          user_tokens: (B, U_max, user_dim) 其中最后一维 user_dim 中包含 active
        """
        B = state_vec.size(0)
        idx = 0

        ru_tokens = state_vec[:, idx: idx + self.n_ru * self.ru_dim].view(B, self.n_ru, self.ru_dim)
        idx += self.n_ru * self.ru_dim

        pn_tokens = state_vec[:, idx: idx + self.n_pn * self.pn_dim].view(B, self.n_pn, self.pn_dim)
        idx += self.n_pn * self.pn_dim

        user_tokens = state_vec[:, idx: idx + self.U_max * self.user_dim].view(B, self.U_max, self.user_dim)
        return ru_tokens, pn_tokens, user_tokens

    def forward(self, state_vec):
        ru_tokens, pn_tokens, user_tokens = self._parse_state(state_vec)
        B = state_vec.size(0)

        # 用户 mask：active=1 表示有效 token
        user_active = user_tokens[:, :, 2] > 0.5  # (B, U_max) bool

        h_ru = self.ru_proj(ru_tokens) + self.type_emb[0].view(1, 1, -1)      # (B,M,d)
        h_pn = self.pn_proj(pn_tokens) + self.type_emb[1].view(1, 1, -1)      # (B,N,d)
        h_u = self.user_proj(user_tokens) + self.type_emb[2].view(1, 1, -1)   # (B,U,d)

        tokens = torch.cat([h_ru, h_pn, h_u], dim=1)  # (B, M+N+U, d)

        # key_padding_mask：True 表示被 mask 掉
        pad_mask = ~user_active  # (B,U_max)
        key_padding_mask = torch.cat(
            [
                torch.zeros(B, self.n_ru + self.n_pn, device=state_vec.device, dtype=torch.bool),
                pad_mask
            ],
            dim=1
        )

        h = self.encoder(tokens, src_key_padding_mask=key_padding_mask)  # (B,seq,d)

        # 切分回各实体 token 输出
        h_ru_out = h[:, :self.n_ru, :]
        h_pn_out = h[:, self.n_ru:self.n_ru + self.n_pn, :]
        h_u_out = h[:, self.n_ru + self.n_pn:, :]

        # Value：masked pooling（RU+PN 永远有效，User 用 active mask）
        valid_mask = torch.cat(
            [
                torch.ones(B, self.n_ru + self.n_pn, device=state_vec.device),
                user_active.float()
            ],
            dim=1
        )  # (B,seq)
        pooled = (h * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)

        value = self.value_head(pooled).squeeze(-1)  # (B,)

        # RU：Bernoulli logit
        ru_logits = self.ru_head(h_ru_out).squeeze(-1)  # (B,M)

        # PN：Beta 分布参数
        pn_ai_alpha = self.softplus(self.pn_ai_alpha_head(h_pn_out)).squeeze(-1) + 1e-3
        pn_ai_beta = self.softplus(self.pn_ai_beta_head(h_pn_out)).squeeze(-1) + 1e-3
        pn_bbu_alpha = self.softplus(self.pn_bbu_alpha_head(h_pn_out)).squeeze(-1) + 1e-3
        pn_bbu_beta = self.softplus(self.pn_bbu_beta_head(h_pn_out)).squeeze(-1) + 1e-3

        return ru_logits, pn_ai_alpha, pn_ai_beta, pn_bbu_alpha, pn_bbu_beta, value


# --- 2. A2C Agent ---
class A2CAgent:
    def __init__(self, env: EnergyEfficiencyEnv, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        self.n_ru = env.n_ru
        self.n_pn = env.n_pn
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy = TransformerActorCritic(
            state_dim=env.state_dim,
            n_ru=env.n_ru,
            n_pn=env.n_pn,
            U_max=env.U_max,
            ru_dim=env.ru_dim,
            pn_dim=env.pn_dim,
            user_dim=env.user_dim,
            d_model=128,
            nhead=8,
            num_layers=2,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_value(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, _, _, _, _, value = self.policy(state_t)
        return float(value.item())

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)  # (1,state_dim)
        with torch.no_grad():
            ru_logits, pn_ai_alpha, pn_ai_beta, pn_bbu_alpha, pn_bbu_beta, value = self.policy(state_t)

            # 1) RU：Bernoulli 采样 0/1
            ru_dist = Bernoulli(logits=ru_logits)  # (1,M)
            ru_actions = ru_dist.sample().squeeze(0)  # (M,)
            ru_logprob = ru_dist.log_prob(ru_actions).sum().item()

            # 2) PN：Beta 采样比例
            pn_ai_dist = Beta(pn_ai_alpha, pn_ai_beta)  # (1,N)
            pn_ai_ratios = pn_ai_dist.sample().squeeze(0)  # (N,)
            pn_ai_logprob = pn_ai_dist.log_prob(pn_ai_ratios).sum().item()

            pn_bbu_dist = Beta(pn_bbu_alpha, pn_bbu_beta)  # (1,N)
            pn_bbu_ratios = pn_bbu_dist.sample().squeeze(0)  # (N,)
            pn_bbu_logprob = pn_bbu_dist.log_prob(pn_bbu_ratios).sum().item()

            total_logprob = ru_logprob + pn_ai_logprob + pn_bbu_logprob
            action = torch.cat([ru_actions, pn_ai_ratios, pn_bbu_ratios], dim=0).cpu().numpy()

        return action, total_logprob, float(value.item())

    def update(self, states, actions, rewards, next_values, dones):
        # states/actions 在这里是 list，元素通常是 numpy.ndarray；
        # 直接 torch.FloatTensor(list_of_ndarray) 会走慢路径，性能警告也由此产生。
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_values = np.asarray(next_values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool_)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_values = torch.from_numpy(next_values)
        dones = torch.from_numpy(dones)

        ru_actions = actions[:, :self.n_ru]
        pn_ai_ratios = actions[:, self.n_ru: self.n_ru + self.n_pn]
        pn_bbu_ratios = actions[:, self.n_ru + self.n_pn:]

        # 前向：得到当前分布参数与价值预测
        ru_logits, pn_ai_alpha, pn_ai_beta, pn_bbu_alpha, pn_bbu_beta, values = self.policy(states)

        # TD target & advantages
        td_target = rewards + self.gamma * next_values * (~dones)
        advantages = td_target - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # 重新计算 logprob
        ru_dist = Bernoulli(logits=ru_logits)
        new_ru_logprobs = ru_dist.log_prob(ru_actions).sum(dim=1)

        pn_ai_dist = Beta(pn_ai_alpha, pn_ai_beta)
        new_pn_ai_logprobs = pn_ai_dist.log_prob(pn_ai_ratios).sum(dim=1)

        pn_bbu_dist = Beta(pn_bbu_alpha, pn_bbu_beta)
        new_pn_bbu_logprobs = pn_bbu_dist.log_prob(pn_bbu_ratios).sum(dim=1)

        new_logprobs = new_ru_logprobs + new_pn_ai_logprobs + new_pn_bbu_logprobs
        entropy = (
            ru_dist.entropy().sum(dim=1)
            + pn_ai_dist.entropy().sum(dim=1)
            + pn_bbu_dist.entropy().sum(dim=1)
        )

        actor_loss = -(new_logprobs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()


# --- 3. 主训练循环 ---
def main():
    print("初始化环境...")
    env = EnergyEfficiencyEnv()
    n_ru = env.n_ru
    n_pn = env.n_pn

    agent = A2CAgent(env, lr=1e-4, gamma=0.99, entropy_coef=0.01)
    episodes = 2000
    ee_history = []

    # 简单动作维度校验
    state = env.reset()
    action, _, _ = agent.select_action(state)
    assert len(action) == env.action_dim, f"动作维度错误：期望{env.action_dim}，实际{len(action)}"
    print(f"动作维度校验通过：{len(action)}")

    power_ru_history = []
    power_pn_history = []
    print("开始训练 (A2C + Transformer)...")
    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # 多步 rollout（匹配 env.max_steps）
        done = False
        while not done:
            action, logprob, _ = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state

            if done:
                # 使用 episode 内最后一步 info 统计功耗（你也可以改成平均）
                power_ru_history.append(info.get("power_ru", 0.0))
                power_pn_history.append(info.get("power_pn", 0.0))

        episode_reward = float(np.mean(rewards))
        ee_history.append(episode_reward)

        # 计算 next_values（done 时为 0）
        next_values = []
        for i in range(len(states)):
            if dones[i]:
                next_values.append(0.0)
            else:
                next_values.append(agent.get_value(next_states[i]))

        # A2C 更新（advantages = td_target - value）
        agent.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_values=next_values,
            dones=dones,
        )

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(ee_history[-100:])
            avg_power_ru = np.mean(power_ru_history[-100:]) if len(power_ru_history) >= 100 else np.mean(power_ru_history)
            avg_power_pn = np.mean(power_pn_history[-100:]) if len(power_pn_history) >= 100 else np.mean(power_pn_history)
            print(f"Episode {episode+1}/{episodes}, Avg EE (last 100): {avg_reward:.5f}")
            print(
                f"Episode {episode + 1}/{episodes}, "
                f"Avg EE: {avg_reward:.5f}, "
                f"Avg Power RU: {avg_power_ru:.2f}W, "
                f"Avg Power PN: {avg_power_pn:.2f}W"
            )
    # --- 可视化 ---
    plt.figure(figsize=(10, 5))
    # 原始曲线 + 更光滑的平滑曲线（MA/EMA）
    ee_arr = np.asarray(ee_history, dtype=np.float32)
    plt.plot(ee_arr, alpha=0.25, linewidth=1.0, label='EE (raw)')

    # 1) 简单滑动平均（窗口越大越光滑，但滞后更明显）
    window = 200  # 想更光滑：改为 300/500；想更灵敏：改为 50/100
    if ee_arr.size >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        ee_ma = np.convolve(ee_arr, kernel, mode='valid')
        x_ma = np.arange(window - 1, ee_arr.size)
        plt.plot(x_ma, ee_ma, linewidth=2.2, label=f'EE (MA{window})')

    # 2) EMA（指数滑动平均，更光滑且相对更“跟手”）
    ema_alpha = 0.03  # 越小越光滑：0.01；越大越敏感：0.05~0.1
    ema = np.empty_like(ee_arr)
    ema[0] = ee_arr[0] if ee_arr.size > 0 else 0.0
    for i in range(1, ee_arr.size):
        ema[i] = ema_alpha * ee_arr[i] + (1.0 - ema_alpha) * ema[i - 1]
    plt.plot(ema, linewidth=2.2, label=f'EE (EMA α={ema_alpha})')
    plt.title('A2C Training Progress (Energy Efficiency)')
    plt.xlabel('Episode')
    plt.ylabel('EE Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('a2c_training.png')
    plt.show()
    np.save("ee_history_drl.npy", ee_arr)
    print("已保存 DRL EE 曲线到 ee_history_drl.npy")

    # --- 标准评估：多次采样统计 ---
    print("\n--- 标准评估 (eval_episodes=100) ---")
    eval_episodes = 100
    eval_rewards = []
    representative_action = None

    for ep in range(eval_episodes):
        state = env.reset()
        done = False
        ep_rewards = []
        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            ep_rewards.append(reward)
            state = next_state
            if ep == eval_episodes - 1 and done:
                representative_action = action
        eval_rewards.append(float(np.mean(ep_rewards)))

    eval_arr = np.asarray(eval_rewards, dtype=np.float32)
    print(f"评估EE均值: {eval_arr.mean():.6f}")
    print(f"评估EE标准差: {eval_arr.std():.6f}")
    print(f"评估EE最小值: {eval_arr.min():.6f}")
    print(f"评估EE最大值: {eval_arr.max():.6f}")

    # 打印一组代表性动作配置（最后一次评估回合的最终动作）
    if representative_action is not None:
        action = representative_action
        print("RU开关状态:", action[:n_ru].astype(int))

        pn_ai_ratios = action[n_ru: n_ru + n_pn]
        pn_bbu_ratios = action[n_ru + n_pn:]
        pn_c_max = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}
        print("\n=== PN分配详情（代表性样本）===")
        for i in range(n_pn):
            pn_id = i + 1
            ai_ratio = pn_ai_ratios[i]
            bbu_ratio = pn_bbu_ratios[i]
            c_max = pn_c_max[pn_id]

            c_ai = max(0, int(round(ai_ratio * c_max)))
            c_bbu = max(0, int(round(bbu_ratio * c_max)))
            total = c_ai + c_bbu
            if total > c_max:
                scale = c_max / total
                c_ai = int(round(c_ai * scale))
                c_bbu = int(round(c_bbu * scale))
                if c_ai + c_bbu > c_max:
                    c_bbu = c_max - c_ai

            sta = 1 if (c_ai > 0 and c_bbu > 0) else 0
            ratio_sum = ai_ratio + bbu_ratio
            print(f"PN{pn_id} (C_max={c_max}):")
            print(f"  原始比例 → C_AI={ai_ratio:.6f}, C_BBU={bbu_ratio:.6f} (比例和={ratio_sum:.6f})")
            print(f"  约束后值 → C_AI={c_ai}, C_BBU={c_bbu} (总和={c_ai + c_bbu} ≤ {c_max}: {c_ai + c_bbu <= c_max})")
            print(f"  PN状态 → {'激活' if sta else '休眠'}")
            print("  ---")


if __name__ == '__main__':
    main()
