# -*- coding: utf-8 -*-
"""图 5-3 按类去重的作用随候选数的变化（CIFAR-100 档）

数据出处：dinov2-route-exp/results_dedup_kscan_c100.json（exp97_dedup_kscan_c100.py）
设定：CIFAR-100，200 节点各 2 类，M=400，每类 4 个副本，只用 384 维原特征。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(ROOT), "dinov2-route-exp", "results")

INK = "#1F2D3A"
SUB = "#4E5F6E"
BLUE = "#35617F"
RED = "#B04A32"
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
})


def tidy(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(INK)
    ax.grid(False)
    ax.tick_params(length=2.5, width=0.6)


def main():
    d = json.load(open(os.path.join(EXP, "results_dedup_kscan_c100.json"),
                       encoding="utf-8"))
    rows = [r for r in d["rows"] if r["k"] <= 10]
    ks = [r["k"] for r in rows]
    ded = [r["dedup"] for r in rows]
    raw = [r["raw"] for r in rows]

    fig, ax = plt.subplots(figsize=(4.6, 2.8))

    ax.plot(ks, raw, "-o", color=RED, lw=1.6, ms=4.2, mfc="white", mew=1.4,
            label="不去重")
    ax.plot(ks, ded, "-o", color=BLUE, lw=1.9, ms=4.2, mfc="white", mew=1.4,
            label="按类去重")

    ax.axvline(3, color=GRY, ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.text(3.12, 0.972, "默认名额 3", fontsize=7.4, color=SUB, va="bottom")

    i3 = ks.index(3)
    ax.annotate(f"{raw[i3]:.4f}", (3, raw[i3]), xytext=(8, -15),
                textcoords="offset points", fontsize=7.4, color=RED)
    ax.annotate(f"{ded[i3]:.4f}", (3, ded[i3]), xytext=(8, -15),
                textcoords="offset points", fontsize=7.4, color=BLUE)

    ax.set_xlabel("交回的候选数")
    ax.set_ylabel("真类落在候选中的比例")
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.79, 1.0)
    tidy(ax)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"fig5_3_dedup.{ext}"), dpi=300)
    plt.close(fig)
    print("saved fig5_3_dedup")


if __name__ == "__main__":
    main()
