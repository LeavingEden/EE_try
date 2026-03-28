import numpy as np
import matplotlib.pyplot as plt

from env_ee import EnergyEfficiencyEnv
from train import A2CAgent


def ema_smooth(arr, alpha=0.03):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def train_for_user_count(user_count, t_min=50, episodes=1000):
    """
    固定用户规模训练：
    - 通过 u_max=user_count 且 user_min=user_max=user_count 固定每步活跃用户数
    - 记录每个 episode 的平均 EE
    """
    env = EnergyEfficiencyEnv(
        u_max=user_count,
        user_min=user_count,
        user_max=user_count,
        t_min=t_min,
    )
    agent = A2CAgent(env, lr=1e-4, gamma=0.99, entropy_coef=0.01)

    ee_history = []
    for ep in range(episodes):
        state = env.reset()
        done = False

        states, actions, rewards, next_states, dones = [], [], [], [], []
        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state

        # episode 平均 EE
        ee_history.append(float(np.mean(rewards)))

        next_values = []
        for i in range(len(states)):
            if dones[i]:
                next_values.append(0.0)
            else:
                next_values.append(agent.get_value(next_states[i]))

        agent.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_values=next_values,
            dones=dones,
        )

        if (ep + 1) % 100 == 0:
            avg = np.mean(ee_history[-100:])
            print(
                f"[Users={user_count}, t_min={t_min}] "
                f"Episode {ep+1}/{episodes}, Avg EE(last100): {avg:.5f}"
            )

    return np.asarray(ee_history, dtype=np.float32)


def main():
    # 4 组训练配置：
    # 1) users=100, t_min=50
    # 2) users=150, t_min=50
    # 3) users=200, t_min=50
    # 4) users=200, t_min=60
    experiments = [
        {"users": 100, "t_min": 50, "label": "Users=100, t_min=50"},
        {"users": 150, "t_min": 50, "label": "Users=150, t_min=50"},
        {"users": 200, "t_min": 50, "label": "Users=200, t_min=50"},
        {"users": 200, "t_min": 60, "label": "Users=200, t_min=60"},
    ]
    episodes = 1000
    ema_alpha = 0.03

    curves = []
    for exp in experiments:
        users = exp["users"]
        t_min = exp["t_min"]
        label = exp["label"]
        print(f"\n=== 开始训练：{label} ===")
        hist = train_for_user_count(user_count=users, t_min=t_min, episodes=episodes)
        curves.append(
            {
                "label": label,
                "ema": ema_smooth(hist, alpha=ema_alpha),
            }
        )

    # 仅绘制 EE(EMA)，不同用户数不同颜色
    plt.figure(figsize=(11, 6))
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, item in enumerate(curves):
        plt.plot(
            item["ema"],
            linewidth=2.2,
            color=color_list[i % len(color_list)],
            label=f"{item['label']} | EE (EMA)",
        )

    plt.title("Transformer-A2C Training Curves under Different User Counts")
    plt.xlabel("Episode")
    plt.ylabel("EE (EMA)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("a2c_training_multi_users_ema.png")
    plt.show()
    print("已保存对比图: a2c_training_multi_users_ema.png")


if __name__ == "__main__":
    main()
