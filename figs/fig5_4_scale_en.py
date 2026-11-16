# -*- coding: utf-8 -*-
"""Fig. 6 (English): effect of the near-tier quota on three-candidate coverage
and cross-class edge ratio. Data identical to fig5_4 in build_data_figs.py
(exp52_scale_route.py, K=1000, R=35, M=35000, L=16)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

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
    "legend.fontsize": 7.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def tidy(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(INK)
    ax.grid(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "en", f"{name}.{ext}"), dpi=300)


def main():
    kn = [2, 4, 6, 8, 10]
    a3 = [0.787, 0.810, 0.830, 0.818, 0.786]
    cross = [0.800, 0.671, 0.598, 0.495, 0.388]

    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    ax.plot(kn, a3, marker="o", ms=4.0, lw=1.6, color=BLUE,
            label=r"$\alpha_3$ (three-candidate coverage)")
    ax.plot(kn, cross, marker="s", ms=3.8, lw=1.6, color=RED,
            label="Cross-class edge ratio")
    for x, y in zip(kn, a3):
        ax.text(x, y + 0.020, f"{y:.3f}", ha="center", fontsize=7.0, color=BLUE)
    for x, y in zip(kn, cross):
        ax.text(x, y - 0.040, f"{y:.3f}", ha="center", fontsize=7.0, color=RED)
    ax.axvline(6, color=GRY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlabel(r"Near-band quota $L_{\mathrm{n}}$ ($L=16$)")
    ax.set_ylabel("Ratio to ceiling\nand cross-class edge ratio")
    ax.set_xticks(kn)
    ax.set_xlim(1.4, 10.6)
    ax.set_ylim(0.32, 0.92)
    tidy(ax)
    ax.legend(loc="lower left", handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig5_4_scale")
    print("saved en/fig5_4_scale")


if __name__ == "__main__":
    main()
