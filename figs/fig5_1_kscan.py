# -*- coding: utf-8 -*-
"""图 5-1：端到端精度随类数的变化（单图，只画三条曲线）

数据出处
  K=10   表 5-1（CIFAR-10，20 节点 × 2 类，每类 4 副本）
  K=20    dinov2-route-exp/results_Kscan20_noW.json
  K=50    dinov2-route-exp/results_Kscan50_noW.json
  K=100  表 5-1（CIFAR-100，200 节点 × 2 类）
集中式一律取闭式解（ridge），与表 5-1 同口径。
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


def load_mid(k):
    d = json.load(open(os.path.join(EXP, f"results_Kscan{k}_noW.json"),
                       encoding="utf-8"))
    return (d["acc"]["ridge_closed"], d["acc"]["C"]["mean"],
            d["acc"]["sys"]["mean"], d["acc"]["B"]["mean"])


def main():
    Ks = [10, 20, 50, 100]
    # K=10 与 K=100 取表 5-1 的读数；K=20、50 取本轮实验
    cent = {10: 0.9692, 100: 0.8230}
    fed = {10: 0.9645, 100: 0.7651}
    ours = {10: 0.9552, 100: 0.8030}
    loc = {10: 0.1990, 100: 0.0198}
    for k in (20, 50):
        c, f, o, b = load_mid(k)
        cent[k], fed[k], ours[k], loc[k] = c, f, o, b

    fig, ax = plt.subplots(figsize=(4.3, 2.75))

    x = np.arange(len(Ks))
    ax.plot(x, [cent[k] for k in Ks], color=GRY, marker="s", ms=4.2, lw=1.5,
            mfc="white", mew=1.1, ls="-", label="集中式（闭式解）")
    ax.plot(x, [fed[k] for k in Ks], color=GRN, marker="^", ms=4.4, lw=1.5,
            mfc="white", mew=1.1, ls="--", label="FedAvg")
    ax.plot(x, [ours[k] for k in Ks], color=BLUE, marker="o", ms=4.8, lw=1.9,
            mfc="white", mew=1.3, ls="-", label="ProtoAtlas")

    for k in Ks:
        off = (11, -9) if k == Ks[0] else (0, -13)
        ax.annotate(f"{ours[k]:.3f}", (x[Ks.index(k)], ours[k]),
                    xytext=off, textcoords="offset points", ha="center",
                    fontsize=7.0, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("全网类别数 $K$")
    ax.set_ylabel("端到端精度")
    ax.set_ylim(0.74, 0.995)
    tidy(ax)
    ax.legend(loc="upper right", handlelength=2.0, borderaxespad=0.4)

    save(fig, "fig5_1_e2e")


if __name__ == "__main__":
    main()
