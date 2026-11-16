# -*- coding: utf-8 -*-
"""临时对照图（不进论文）：CIFAR-10 与 CIFAR-100 两档上去重前后的召回曲线"""
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def load(name, kmax=10):
    d = json.load(open(os.path.join(EXP, name), encoding="utf-8"))
    rows = [r for r in d["rows"] if r["k"] <= kmax]
    return ([r["k"] for r in rows], [r["raw"] for r in rows],
            [r["dedup"] for r in rows], d["diff3"])


def main():
    k10, raw10, ded10, d10 = load("results_dedup_kscan.json")
    k100, raw100, ded100, d100 = load("results_dedup_kscan_c100.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

    for ax, ks, raw, ded, title, dv in [
            (axes[0], k10, raw10, ded10, f"CIFAR-10（10 类，40 vnode）", d10),
            (axes[1], k100, raw100, ded100, f"CIFAR-100（100 类，400 vnode）", d100)]:
        ax.plot(ks, raw, "-o", color=RED, lw=1.6, ms=3.8, mfc="white", mew=1.3,
                label="不去重")
        ax.plot(ks, ded, "-o", color=BLUE, lw=1.9, ms=3.8, mfc="white", mew=1.3,
                label="按类去重")
        ax.axvline(3, color=GRY, ls=(0, (4, 3)), lw=0.9, zorder=0)
        ax.set_title(title, fontsize=9, color=INK, pad=6)
        ax.set_xlabel("交回的候选数")
        ax.set_xticks(ks)
        ax.set_xlim(0.6, 10.4)
        ax.text(0.04, 0.06, f"前 3 名平均含 {dv:.2f} 个不同类",
                transform=ax.transAxes, fontsize=7.6, color=SUB)
        tidy(ax)

    axes[0].set_ylabel("真类落在候选中的比例")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "tmp_cmp_dedup.png"), dpi=300)
    print("saved tmp_cmp_dedup.png")


if __name__ == "__main__":
    main()
