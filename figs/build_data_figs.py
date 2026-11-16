# -*- coding: utf-8 -*-
"""生成第 5、6 章的定量图（数据取自 dinov2-route-exp/results_*.json）

输出：每图一份 PDF（矢量，投稿用）+ 一份 PNG（3x，预览用）
版式约定：单栏宽 17 cm，最小字号 7.5 pt，中文字体 Microsoft YaHei
图内不加标题（中文期刊图题在图下方，由正文给出）
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
ORANGE = "#BE8560"
CLS = ["#CFE0EE", "#F3DECF", "#DCEADA"]

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

W1 = 6.7  # 单栏 17 cm


def fig5_dist():
    """图 5-2：多起点与一跳扩散的召回—代价权衡

    实现与数据出处见 fig5_2_pareto.py，数据取自
    dinov2-route-exp/results_greedy_grid_lda.json（tag=lda，d=384）"""
    import fig5_2_pareto
    fig5_2_pareto.main()


def fig5_dedup():
    """图 5-3 按类去重的作用随候选数的变化（CIFAR-100 档，数据见 exp97_dedup_kscan_c100.py）"""
    import fig5_3_dedup
    fig5_3_dedup.main()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("saved", name)


def load(fn):
    with open(os.path.join(EXP, fn), encoding="utf-8") as f:
        return json.load(f)


def tidy(ax, grid_axis="y"):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color(INK)
    ax.grid(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 图 C-1
def figC_1():
    """端到端精度的三项分解：柱顶为 1，蓝段即 E2E"""
    rows = [
        ("CIFAR-10\nDINOv2 特征", 0.9692, 0.9682, 0.9987),
        ("CIFAR-10\n原始像素", 0.3666, 0.3748, 0.8850),
        ("CIFAR-100\nDINOv2 特征", 0.8308, 0.8237, 0.9994),
    ]
    labels, hit, exp_loss, route_loss, gpts = [], [], [], [], []
    for name, g, a, s in rows:
        labels.append(name)
        gpts.append(g)
        hit.append(a * s)
        exp_loss.append(a * (1 - s))
        route_loss.append((1 - a))

    x = np.arange(len(rows))
    w = 0.42
    fig, ax = plt.subplots(figsize=(W1, 2.75))

    ax.bar(x, hit, w, color=BLUE, label="命中且专家答对  $A\\cdot S_{\\exp}$")
    ax.bar(x, exp_loss, w, bottom=hit, color="#D9A441",
           label="专家损耗  $A(1-S_{\\exp})$")
    bot3 = np.array(hit) + np.array(exp_loss)
    ax.bar(x, route_loss, w, bottom=bot3, color=CLS[0], hatch="///",
           edgecolor=BLUE, lw=0.6, label="路由损耗  $(1-A)$")

    for i, (h, e, r) in enumerate(zip(hit, exp_loss, route_loss)):
        ax.text(x[i], h / 2, f"{h:.3f}", ha="center", va="center",
                color="white", fontsize=8.5)
        if r > 0.05:
            ax.text(x[i], h + e + r / 2, f"{r:.3f}", ha="center", va="center",
                    color=BLUE, fontsize=8.5)
        ax.plot([x[i] - w / 2, x[i] + w / 2], [gpts[i]] * 2, color=RED, lw=1.4)
        ax.text(x[i] + w / 2 + 0.04, gpts[i], f"$G$={gpts[i]:.4f}",
                color=RED, fontsize=8, va="center")


    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("样本占比")
    ax.set_ylim(0, 1.10)
    ax.set_xlim(-0.6, 2.85)
    tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              handlelength=1.2, columnspacing=1.5, labelspacing=0.3)
    save(fig, "figC_1_decomp")


# ---------------------------------------------------------------- 图 A-1
def figA_1():
    """候选召回随坐标维度与类数的变化"""
    d = load("results_K_curve.json")
    ks = d["ks"]
    dims = [8, 16, 32, 64, 128, 256, 384]
    colors = ["#9DB0BF", "#7FA3BC", BLUE, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(W1, 2.75))
    for k, c in zip(ks, colors):
        ys = [d["table"][str(dd)][ks.index(k)] for dd in dims]
        ax.plot(dims, ys, marker="o", ms=3.2, lw=1.3, color=c, label=f"$K$={k}")
    for k, dy in ((100, 0.0), (50, -0.020)):
        ys = [d["table"][str(dd)][ks.index(k)] for dd in dims]
        ax.text(410, ys[-1] + dy, f"{ys[-1]:.3f}", color=RED if k == 100 else ORANGE,
                fontsize=7.5, va="center")

    # 0.95 是 d*（达到该召回的最小维度）的判定阈值，不是别的量
    ax.axhline(0.95, color=GRY, ls="--", lw=0.9)
    ax.text(9.5, 0.775, "虚线为 $d^{*}$ 的判定阈值 0.95", color=GRY, fontsize=7.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(v) for v in dims])
    ax.set_xlabel("坐标维度 $d$")
    ax.set_ylabel("3 候选召回（全局排序口径）")
    ax.set_ylim(0.65, 1.02)
    ax.set_xlim(6.5, 760)
    tidy(ax)
    ax.legend(ncol=5, loc="lower right", handlelength=1.0,
              columnspacing=1.0, fontsize=7.6, bbox_to_anchor=(1.0, -0.02))
    save(fig, "figA_1_recall")


# ---------------------------------------------------------------- 图 5-4
def fig5_4():
    """近档名额对三候选覆盖与跨类边占比的影响（单图，折线）
    exp52_scale_route.py，K=1000，R=35，M=35000，kappa=16"""
    kn = [2, 4, 6, 8, 10]
    a3 = [0.787, 0.810, 0.830, 0.818, 0.786]
    cross = [0.800, 0.671, 0.598, 0.495, 0.388]

    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    ax.plot(kn, a3, marker="o", ms=4.0, lw=1.6, color=BLUE, label=r"$\alpha_3$（三候选覆盖）")
    ax.plot(kn, cross, marker="s", ms=3.8, lw=1.6, color=RED, label="跨类边占比")
    for x, y in zip(kn, a3):
        ax.text(x, y + 0.020, f"{y:.3f}", ha="center", fontsize=7.0, color=BLUE)
    for x, y in zip(kn, cross):
        ax.text(x, y - 0.040, f"{y:.3f}", ha="center", fontsize=7.0, color=RED)
    ax.axvline(6, color=GRY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xlabel(r"近档名额 $L_{\mathrm{n}}$（$L=16$）")
    ax.set_ylabel("相对上限的比值\n与跨类边占比")
    ax.set_xticks(kn)
    ax.set_xlim(1.4, 10.6)
    ax.set_ylim(0.32, 0.92)
    tidy(ax)
    ax.legend(loc="lower left", handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig5_4_scale")



# ---------------------------------------------------------------- 图 5-3
def fig5_e2e():
    """图 5-1：端到端精度随类数的变化

    实现与数据出处见 fig5_1_kscan.py。K=10 与 K=100 取表 5-1 的读数，
    K=20 与 K=50 取自 results_Kscan20_noW.json 与 results_Kscan50_noW.json。"""
    import fig5_1_kscan
    fig5_1_kscan.main()


def figB_1():
    """误识的四象限：占比与各象限内接收节点的答对率"""
    fig, ax = plt.subplots(figsize=(W1, 3.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    c1x, c1w = 1.05, 4.05
    c2x, c2w = 5.45, 4.35
    cells = [
        # (x, y, w, 标题, 占比, 备注, 底色, 边色)
        (c1x, 3.55, c1w, "A　路由正确", "94.22%",
         "接收节点即真实类的持有者\n其判别头答对 99.7%", CLS[0], BLUE),
        (c2x, 3.55, c2w, "—　路由正确而不持有", "0",
         "按定义不存在：路由正确即\n终点 vnode 的类等于真类", "#F4F6F8", GRY),
        (c1x, 0.55, c1w, "B　送错，但持有真实类", "0.54%",
         "两类的判别头都能判\n实测答对 63%", "#F6EBD8", "#C08A2E"),
        (c2x, 0.55, c2w, "C　送错，且不持有真实类", "5.24%",
         "接收节点没有这一类，答对率 0\n受理层只拦住 30.5%", "#F7EAE7", RED),
    ]
    for x, y, w, title, pct, note, fc, ec in cells:
        ax.add_patch(plt.Rectangle((x, y), w, 2.55, facecolor=fc,
                                   edgecolor=ec, lw=0.9, zorder=1))
        ax.text(x + 0.22, y + 2.16, title, fontsize=8.4, color=ec, zorder=2)
        ax.text(x + 0.22, y + 1.30, pct, fontsize=16, color=INK, zorder=2,
                fontweight="bold")
        ax.text(x + 0.22, y + 0.26, note, fontsize=7.5, color=SUB, zorder=2,
                linespacing=1.6)

    ax.text(c1x + c1w / 2, 6.36, "接收节点持有真实类", fontsize=8.2,
            color=SUB, ha="center")
    ax.text(c2x + c2w / 2, 6.36, "接收节点不持有真实类", fontsize=8.2,
            color=SUB, ha="center")
    ax.text(0.50, 4.82, "路\n由\n正\n确", fontsize=8.2, color=SUB,
            ha="center", va="center", linespacing=1.35)
    ax.text(0.50, 1.82, "路\n由\n错\n误", fontsize=8.2, color=SUB,
            ha="center", va="center", linespacing=1.35)
    save(fig, "figB_1_quadrant")


def fig4_2_bytes():
    """一次查询的字节构成：按阶段堆叠，三档跳数"""
    L = 8          # 标识字节
    d = 99         # 坐标维度
    hop = 4 * d + L            # 404 B
    verify = 3 * hop           # k = 3，1212 B
    feat = 1536                # D = 384 单精度
    back = 100                 # 回传上界

    hs = [2, 3, 4]
    segs = [
        ("路由转发（每跳 %d B）" % hop, lambda h: h * hop, "#CFE0EE", BLUE),
        ("候选核实（k = 3，一次）", lambda h: verify, "#F3DECF", "#A9702F"),
        ("判别特征（384 维）", lambda h: feat, "#DCEADA", GRN),
        ("回传（≤ %d B）" % back, lambda h: back, "#E7ECF0", SUB),
    ]

    fig, ax = plt.subplots(figsize=(W1, 2.45))
    ys = [2, 1, 0]
    for y, h in zip(ys, hs):
        left = 0
        for name, fn, fc, ec in segs:
            w = fn(h)
            ax.barh(y, w, left=left, height=0.44, color=fc,
                    edgecolor="white", linewidth=1.0, zorder=2)
            if w >= 700:
                ax.text(left + w / 2, y, f"{w}", ha="center", va="center",
                        fontsize=7.6, color=ec, zorder=3)
            left += w
        ax.text(left + 55, y, f"合计 {left / 1024:.1f} KB", va="center",
                fontsize=7.8, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"$h$ = {h}" for h in hs])
    ax.set_ylim(-0.52, 2.42)
    ax.set_xlim(0, 5250)
    ax.set_xticks([0, 1024, 2048, 3072, 4096])
    ax.set_xticklabels(["0", "1", "2", "3", "4 KB"])
    ax.set_xlabel("一次查询离开源节点的字节数", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="white")
               for _, _, fc, _ in segs]
    ax.legend(handles, [s[0] for s in segs], loc="lower center",
              bbox_to_anchor=(0.42, 1.0), ncol=4, handlelength=1.1,
              handleheight=0.9, columnspacing=1.4, fontsize=7.6)

    save(fig, "fig4_2_bytes")


if __name__ == "__main__":
    fig4_2_bytes()
    fig5_e2e()
    fig5_dist()
    fig5_dedup()
    fig5_4()
    figC_1()
    figA_1()
    figB_1()
