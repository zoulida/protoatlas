"""分析 exp87：body 在簇内共同训练能不能更快或更准

读 dinov2-route-exp/results_body_sync_speed_*.json，打印汇总表并画
fig_body_sync.png（四格：同步频率、早停、无增益的两档、代价—增益）。

档位记号：
  L   本地训练，body 不交换
  S2/S10/S50  每 2/10/50 步在 8 个坐标最近节点间平均 body
  E40 每 10 步同步，但只在第 40 步之前，之后纯本地
  R10 每 10 步与 8 个随机节点平均（对照，检验邻近性是否必要）
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "dinov2-route-exp", "results")
OUT = os.path.join(HERE, "fig_body_sync")
BODY_BYTES = 384 * 256 * 4 + 256 * 4

NAME = {"L": "本地训练", "S2": "同步每 2 步", "S10": "同步每 10 步",
        "S50": "同步每 50 步", "E40": "前 40 步同步", "R10": "随机节点同步",
        "W1": "本地训练，权重衰减 1e-3", "W5": "本地 + body 正则 5e-3",
        "W10": "本地训练，body 权重衰减 1e-2"}
COLOR = {"L": "#2b2b2b", "S2": "#d94801", "S10": "#2171b5",
         "S50": "#238b45", "E40": "#6a51a3", "R10": "#969696",
         "W1": "#fdae6b", "W5": "#cb181d", "W10": "#8c2d04"}


def load(name):
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def steps_of(d):
    return d["eval_points"][:len(d["arms"]["L"]["curve_mean"])]


def comm_mb(d, tag):
    r = d["arms"][tag]
    return r["rounds"] * 8 * 2 * BODY_BYTES / 1e6


def report(d, title):
    st = steps_of(d)
    L = d["arms"]["L"]
    print(f"\n== {title} ==  节点 {d['n_nodes']}，每节点 {d['cls_per_node']} 类，"
          f"训练 {d['train_per_node']} 张")
    print(f"{'档':<5}{'终点':>8}{'±':>8}{'相对本地':>10}{'同步轮':>8}{'通信MB/节点':>12}")
    for k, r in d["arms"].items():
        dv = (r["final"] - L["final"]) * 100
        print(f"{k:<5}{r['final']:8.4f}{r['final_std']:8.4f}{dv:+9.2f} "
              f"{r['rounds']:8d}{comm_mb(d, k):12.0f}")
    # 首次达到本地终点的步数
    for k, r in d["arms"].items():
        hit = next((s for s, a in zip(st, r["curve_mean"]) if a >= L["final"]), None)
        print(f"    {k}: 首次达到本地终点的精度，步数 {hit}")


def graft(dst, src, tag):
    """把只跑了本地一档的结果挂到 dst 上，改名为 tag"""
    import copy
    r = copy.deepcopy(src["arms"]["L"])
    r["rounds"] = 0
    dst["arms"][tag] = r


def panel(ax, d, tags, title, ylim=None):
    st = steps_of(d)
    for k in tags:
        r = d["arms"][k]
        y = np.array(r["curve_mean"])
        sd = np.array(r["curve_std"])
        ax.plot(st, y, marker="o", ms=2.6, lw=1.5, color=COLOR[k], label=NAME[k])
        ax.fill_between(st, y - sd, y + sd, color=COLOR[k], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("本地训练步数", fontsize=8.5)
    ax.set_ylabel("节点本地测试精度", fontsize=8.5)
    ax.set_title(title, fontsize=9.2, pad=4)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.22, lw=0.5)
    ax.tick_params(labelsize=7.6)
    ax.legend(fontsize=6.8, loc="lower right", frameon=True, facecolor="white",
              framealpha=0.88, edgecolor="none", borderpad=0.4)


def main():
    scarce = load("results_body_sync_speed_1k_pc5.json")
    early = load("results_body_sync_speed_1k_pc5_early40.json")
    rand = load("results_body_sync_speed_1k_pc5_rand.json")
    rich = load("results_body_sync_speed_1k_pc25.json")
    c100 = load("results_body_sync_speed_c100_pc25.json")

    report(scarce, "ImageNet-1k，每节点 20 类 × 5 张")
    report(early, "ImageNet-1k，每节点 20 类 × 5 张（早停档）")
    report(rand, "ImageNet-1k，每节点 20 类 × 5 张（随机邻居）")
    report(rich, "ImageNet-1k，每节点 20 类 × 25 张")
    report(c100, "CIFAR-100，每节点 2 类 × 125 张")

    # 正则化对照：只跑本地一档，改权重衰减
    graft(scarce, load("results_body_sync_speed_1k_pc5_wd0.001.json"), "W1")
    graft(scarce, load("results_body_sync_speed_1k_pc5_wd0.005_body.json"), "W5")
    try:
        graft(scarce, load("results_body_sync_speed_1k_pc5_wd0.01_body.json"), "W10")
    except FileNotFoundError:
        pass
    graft(rich, load("results_body_sync_speed_1k_pc25_wd0.005_body.json"), "W5")
    for k in [k for k in scarce["arms"] if k.startswith("W")]:
        r = scarce["arms"][k]
        print(f"  {NAME[k]}: {r['final']:.4f}，相对本地 "
              f"{(r['final']-scarce['arms']['L']['final'])*100:+.2f} 点，零通信")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))

    panel(axes[0, 0], scarce, ["L", "S2", "S10", "S50", "W5"],
          "(a) 数据稀缺：同步频率与本地正则的对比", (0.72, 0.87))
    axes[0, 0].text(0.03, 0.94, "同步档在 30 至 40 步见顶后回落，\n本地正则档持续上升",
                    transform=axes[0, 0].transAxes, fontsize=6.8, va="top",
                    color="#555555")
    panel(axes[0, 1], early, ["L", "S10", "E40"],
          "(b) 只在前 40 步同步，之后纯本地", (0.72, 0.87))
    panel(axes[1, 0], rich, ["L", "S10", "W5"],
          "(c) 数据充足：每节点 500 张、20 类", (0.86, 0.90))
    panel(axes[1, 1], c100, ["L", "S10"],
          "(d) 正文设定：CIFAR-100 每节点 2 类", (0.988, 0.996))

    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")
    print("\n图写入", OUT + ".png")

    # 代价—增益
    print("\n== 代价与增益（每节点累计通信，按 8 邻居全交换计）==")
    for k in ["S2", "S10", "S50"]:
        print(f"  {k}: {comm_mb(scarce, k):.0f} MB，增益 "
              f"{(scarce['arms'][k]['final'] - scarce['arms']['L']['final'])*100:+.2f} 点")
    print(f"  E40: {comm_mb(early, 'E40'):.0f} MB，增益 "
          f"{(early['arms']['E40']['final'] - early['arms']['L']['final'])*100:+.2f} 点")
    print(f"  R10: {comm_mb(rand, 'R10'):.0f} MB，增益 "
          f"{(rand['arms']['R10']['final'] - rand['arms']['L']['final'])*100:+.2f} 点")


if __name__ == "__main__":
    main()
