# -*- coding: utf-8 -*-
"""图 5-2：多起点的召回—代价权衡（384 维坐标，K=100，只取 1 跳档）

数据出处：dinov2-route-exp/results_greedy_grid_lda.json（tag=lda）
每个方案用独立随机流，重跑逐位相同。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(ROOT), "dinov2-route-exp", "results")

INK = "#1F2D3A"
SUB = "#4E5F6E"
LINE = "#9DB0BF"
BLUE = "#35617F"
RED = "#B04A32"
GRN = "#4A7A5C"
GRY = "#8A9AA6"

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.edgecolor": SUB,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 7.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

W = 6.7


def tidy(ax, grid_axis="y"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(INK)
    ax.grid(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"{name}.{ext}"),
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("saved", name)


def load():
    d = json.load(open(os.path.join(EXP, "results_greedy_grid_lda.json"),
                       encoding="utf-8"))["lda"]
    rows = {}
    for k, v in d.items():
        if isinstance(v, dict) and k.startswith("S"):
            ns, hop = k[1:].split("H")
            rows[(int(ns), int(hop))] = v
    return d["global_r3"], d["stuck_frac"], rows


def main():
    upper, stuck, rows = load()
    starts = [1, 2, 3, 5]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))

    # ---------------- (a) 召回—询问量
    ax = axes[0]
    xs = [rows[(s, 1)]["asked"] for s in starts]
    ys = [rows[(s, 1)]["r3"] for s in starts]
    ax.plot(xs, ys, color=BLUE, ls="-", marker="o", ms=4.6, lw=1.6,
            mfc="white", mew=1.2, label="多起点并行下降")

    ax.axhline(upper, color=GRY, lw=1.0, ls=":")
    ax.text(23.6, upper + 0.0012, f"全局排序上界 {upper:.4f}", ha="right",
            va="bottom", fontsize=7.2, color=SUB)

    offs = {1: (-11, 8), 2: (9, -13), 3: (15, -11), 5: (7, -14)}
    for s0, xx, yy in zip(starts, xs, ys):
        ax.annotate(f"{s0} 起点", (xx, yy), xytext=offs[s0],
                    textcoords="offset points", ha="center", fontsize=7.2,
                    color=BLUE)

    ax.set_xlim(12.2, 24.0)
    ax.set_ylim(0.812, 0.940)
    ax.set_xlabel("单次查询询问的 vnode 数")
    ax.set_ylabel("候选召回@3")
    tidy(ax)
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.4)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, fontsize=9, color=SUB,
            fontweight="bold", ha="left", va="top")

    # 右侧差距轴
    ax2 = ax.twinx()
    ax2.set_ylim(*[(upper - y) * 100 for y in ax.get_ylim()])
    ax2.set_ylabel("差距（百分点）", color=GRY, fontsize=8.2, labelpad=2)
    ax2.tick_params(axis="y", colors=GRY, labelsize=8)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(INK)
    ax2.spines["right"].set_linewidth(0.8)

    # ---------------- (b) 边际收益与代价
    ax = axes[1]
    seg = [(1, 2), (2, 3), (3, 5)]
    gain, cost = [], []
    for a, b in seg:
        dg = (rows[(b, 1)]["r3"] - rows[(a, 1)]["r3"]) * 100
        da = rows[(b, 1)]["asked"] - rows[(a, 1)]["asked"]
        gain.append(dg)
        cost.append(da / dg)          # 每多一个百分点召回，多问几个 vnode

    x = np.arange(len(seg))
    bars = ax.bar(x, gain, width=0.5, color=[BLUE, BLUE, "#7C9CB4"],
                  edgecolor="white", lw=0.8)
    for i, v in enumerate(gain):
        ax.text(i, v + 0.18, f"+{v:.2f}", ha="center", fontsize=7.6, color=BLUE)
    ax.set_ylim(0, max(gain) * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}→{b} 起点" for a, b in seg])
    ax.set_ylabel("召回@3 的增量（百分点）")
    tidy(ax)

    ax2 = ax.twinx()
    ax2.plot(x, cost, color=RED, marker="D", ms=4.0, lw=1.4, mfc="white",
             mew=1.1, zorder=3)
    for i, v in enumerate(cost):
        off = (-2, -14) if i == len(cost) - 1 else (15, -3)
        ha = "center" if i == len(cost) - 1 else "left"
        ax2.annotate(f"{v:.2f}", (i, v), xytext=off, textcoords="offset points",
                     ha=ha, fontsize=7.4, color=RED)
    ax2.set_ylim(0, max(cost) * 1.55)
    ax2.set_ylabel("每点召回所需增加的询问数", color=RED, fontsize=8.5)
    ax2.tick_params(axis="y", colors=RED, labelsize=8)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(INK)
    ax2.spines["right"].set_linewidth(0.8)

    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, fontsize=9, color=SUB,
            fontweight="bold", ha="left", va="top")

    fig.subplots_adjust(wspace=0.62)
    save(fig, "fig5_2_dist")


if __name__ == "__main__":
    main()
