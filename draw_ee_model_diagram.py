import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc="#eef4ff", ec="#3b6fb6", fs=13):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"),
    )


def main():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "能效(EE)建模关系图（含时延与功耗）",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    # Latency branch
    add_box(
        ax,
        0.06,
        0.73,
        0.27,
        0.16,
        "用户时延分解\n"
        "T_k = T_{r,k} + T_{p,k} + T_{w,k}\n"
        "T_{r,k}: 无线传输时延\n"
        "T_{p,k}: 处理时延\n"
        "T_{w,k}: 有线回传时延",
        fc="#edf7ee",
        ec="#4b8f58",
        fs=13,
    )

    add_box(
        ax,
        0.06,
        0.49,
        0.27,
        0.14,
        "服务判定\n"
        "I_k = 1(T_k < t_min)\n"
        "N_served = Σ_k I_k",
        fc="#f3fbf4",
        ec="#4b8f58",
        fs=14,
    )
    arrow(ax, 0.195, 0.73, 0.195, 0.63)

    # Power branch
    add_box(
        ax,
        0.39,
        0.73,
        0.25,
        0.16,
        "RU功耗\n"
        "P_m = p_sleep (无用户)\n"
        "P_m = p_fix + |K_m|·P_k/η_pa (有用户)\n"
        "P_RU = Σ_m P_m",
        fc="#fff4ea",
        ec="#c27a2c",
        fs=13,
    )

    add_box(
        ax,
        0.67,
        0.73,
        0.27,
        0.16,
        "PN功耗\n"
        "P_j = P_j^{AI} + P_j^{BBU} + P_j^{sta}\n"
        "休眠态: P_j = p_sleep\n"
        "P_PN = Σ_j P_j",
        fc="#fff4ea",
        ec="#c27a2c",
        fs=13,
    )

    add_box(
        ax,
        0.49,
        0.49,
        0.30,
        0.14,
        "总功耗\n"
        "P_total = P_RU + P_PN\n"
        "= Σ_m P_m + Σ_j P_j",
        fc="#fffaf0",
        ec="#c27a2c",
        fs=14,
    )
    arrow(ax, 0.515, 0.73, 0.56, 0.63)
    arrow(ax, 0.805, 0.73, 0.72, 0.63)

    # EE and reward
    add_box(
        ax,
        0.30,
        0.23,
        0.40,
        0.18,
        "能效指标\n"
        "EE = τ · fps · N_served / P_total\n"
        "   = τ · fps · (Σ_k 1(T_k < t_min)) / (Σ_m P_m + Σ_j P_j)",
        fc="#eaf0ff",
        ec="#3058a8",
        fs=15,
    )
    arrow(ax, 0.195, 0.49, 0.42, 0.41)
    arrow(ax, 0.64, 0.49, 0.58, 0.41)

    add_box(
        ax,
        0.33,
        0.04,
        0.34,
        0.12,
        "奖励函数（训练使用）\n"
        "r = EE - λ_1(1-served_ratio) - λ_2·one_side_alloc - λ_3·1_invalid",
        fc="#f7f0ff",
        ec="#7a4fb0",
        fs=13,
    )
    arrow(ax, 0.50, 0.23, 0.50, 0.16)

    fig.tight_layout()
    fig.savefig("ee_model_with_latency_power.png", dpi=260)
    plt.close(fig)
    print("已生成: ee_model_with_latency_power.png")


if __name__ == "__main__":
    main()
