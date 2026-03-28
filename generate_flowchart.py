import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Windows 下优先使用常见中文字体，避免中文方块
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, w, h, text, fc="#eaf2ff", ec="#2f5da8", fontsize=16):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, p1, p2, color="#444"):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(arrowstyle="->", lw=1.4, color=color),
    )


def add_panel(ax, xy, w, h, title, fc="#f8fbff", ec="#9bb6d9"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
        alpha=0.5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.01, y + h - 0.02, title, ha="left", va="top", fontsize=17, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(19, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.97,
        "基于 Transformer 增强的 A2C 算法完整流程图",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )

    # 分区（减少大方块堆叠，改为“整体+局部”）
    add_panel(ax, (0.04, 0.08), 0.52, 0.84, "整体主流程（System Loop）")
    add_panel(ax, (0.60, 0.54), 0.36, 0.38, "Transformer 子模块")
    add_panel(ax, (0.60, 0.08), 0.36, 0.40, "A2C 子模块")

    # 左侧主流程（只保留关键节点）
    add_box(ax, (0.20, 0.86), 0.20, 0.045, "开始", fc="#fff7e6", ec="#cc7a00")
    add_box(ax, (0.10, 0.78), 0.40, 0.06, "环境初始化 + 动态用户更新")
    add_box(ax, (0.10, 0.68), 0.40, 0.08, "策略推理并执行动作\n(由右侧 Transformer+A2C 子模块提供)")
    add_box(ax, (0.10, 0.56), 0.40, 0.08, "环境Step\n拓扑关联 + 时延/功耗 + EE")
    add_box(ax, (0.10, 0.44), 0.40, 0.08, "奖励整形\nEE + 软惩罚")
    add_box(ax, (0.10, 0.32), 0.40, 0.08, "轨迹缓存\n(s, a, r, s', done)")
    add_box(ax, (0.10, 0.20), 0.40, 0.08, "参数更新\n(调用 A2C 更新)")
    add_box(ax, (0.20, 0.11), 0.20, 0.05, "评估与可视化", fc="#fff3f8", ec="#a63f6d")

    # 主流程箭头
    add_arrow(ax, (0.30, 0.86), (0.30, 0.84))
    add_arrow(ax, (0.30, 0.78), (0.30, 0.76))
    add_arrow(ax, (0.30, 0.68), (0.30, 0.64))
    add_arrow(ax, (0.30, 0.56), (0.30, 0.52))
    add_arrow(ax, (0.30, 0.44), (0.30, 0.40))
    add_arrow(ax, (0.30, 0.32), (0.30, 0.28))
    add_arrow(ax, (0.30, 0.20), (0.30, 0.16))

    # 回路箭头（时间步推进）
    add_arrow(ax, (0.10, 0.60), (0.06, 0.60))
    add_arrow(ax, (0.06, 0.60), (0.06, 0.81))
    add_arrow(ax, (0.06, 0.81), (0.10, 0.81))
    ax.text(0.045, 0.71, "时间步循环", rotation=90, fontsize=15, va="center")

    # Transformer 子模块（局部细化）
    add_box(ax, (0.64, 0.82), 0.28, 0.07, "Token构建\nRU / PN / User + Mask", fc="#edf6ff")
    add_box(ax, (0.64, 0.72), 0.28, 0.07, "Transformer Encoder\n多头自注意力建模全局依赖", fc="#dfefff")
    add_box(ax, (0.64, 0.62), 0.28, 0.07, "特征输出 h", fc="#edf6ff")
    add_arrow(ax, (0.78, 0.82), (0.78, 0.79))
    add_arrow(ax, (0.78, 0.72), (0.78, 0.69))

    # A2C 子模块（局部细化）
    add_box(ax, (0.64, 0.40), 0.28, 0.07, "Actor头\nRU: Bernoulli, PN: Beta", fc="#fff1e6", ec="#cc7a00")
    add_box(ax, (0.64, 0.30), 0.28, 0.07, "Critic头\nV(s)", fc="#fff1e6", ec="#cc7a00")
    add_box(ax, (0.64, 0.20), 0.28, 0.07, "TD Target + Advantage\n(标准化)", fc="#fff8ee", ec="#cc7a00")
    add_box(ax, (0.64, 0.10), 0.28, 0.07, "Loss与更新\nActor + 0.5Critic - Entropy", fc="#ffe9d6", ec="#cc7a00")
    add_arrow(ax, (0.78, 0.40), (0.78, 0.37))
    add_arrow(ax, (0.78, 0.30), (0.78, 0.27))
    add_arrow(ax, (0.78, 0.20), (0.78, 0.17))

    # 分区间连接：主流程 <-> 子模块
    add_arrow(ax, (0.50, 0.72), (0.64, 0.72))  # 主流程调用 Transformer

    # Transformer -> A2C：改为分段折线，避免一整条直线
    add_arrow(ax, (0.78, 0.62), (0.78, 0.53))
    add_arrow(ax, (0.78, 0.53), (0.78, 0.47))

    add_arrow(ax, (0.64, 0.14), (0.50, 0.24))  # A2C 更新反馈到主流程
    ax.text(0.66, 0.16, "更新网络参数 θ", fontsize=15, color="#444")

    # 右下补充说明
    add_box(
        ax,
        (0.62, 0.01),
        0.32,
        0.06,
        "稳定化：Adv标准化 / Entropy Bonus / 软惩罚 / 平缓用户动态",
        fc="#f7fff0",
        ec="#3f8f3f",
        fontsize=15,
    )

    fig.tight_layout()
    fig.savefig("transformer_a2c_flowchart.png", dpi=260)
    plt.close(fig)
    print("已生成: transformer_a2c_flowchart.png")


if __name__ == "__main__":
    main()
