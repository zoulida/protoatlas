"""exp36: DINOv2 在 CIFAR-100 上到底能分开多少类

"DINOv2 最多能分多少类"没有单一答案：
  · 它是无监督特征提取器，训练时没有类别标签（LVD-142M，1.42 亿张）
  · 能分多少类取决于：图像分辨率/域、类间差异、用什么分类头
  · 文献：ImageNet-1k(1000 类) ViT-S/14 kNN 79.0% / linear 81.1%
          iNaturalist-2021(10,000 类) 检索 85.7%

本实验问一个更实用的问题：
  在 CIFAR-100 这 100 个类里，DINOv2 特征**实际能干净分开**的有多少类？
  即：有多少类被"永久吞并"进别的类（无论怎么查都召不回）。

输出：
  ① 每类召回（原型 1-NN），列出 < 0.5 的类
  ② 混淆最严重的类对（A 被认成 B 的双向概率）
  ③ 贪心最大可分子集大小 = "这个特征在这个数据集上实际能承载的类数"
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]

    protos = np.stack([H[y == c].mean(0) for c in range(100)]).astype(np.float32)
    D = dist(T, protos)
    pred = D.argmin(1)

    # 混淆矩阵 100x100（行=真类，列=预测类）
    M = np.zeros((100, 100), np.int32)
    for t, p in zip(yT, pred):
        M[t, p] += 1

    rec = np.diag(M) / M.sum(1)
    log("=" * 78)
    log("① 每类召回（原型 1-NN，全维 384）")
    log("=" * 78)
    log(f"    平均 {rec.mean():.4f}   中位数 {np.median(rec):.4f}   "
        f"最低 {rec.min():.4f}   最高 {rec.max():.4f}")
    for th in [0.9, 0.8, 0.5, 0.2]:
        log(f"    召回 >= {th:.1f} 的类数：{int((rec >= th).sum())} / 100")

    bad = np.argsort(rec)[:8]
    log(f"\n    最差的 8 个类（召回 < 0.5 的共 {int((rec < 0.5).sum())} 个）：")
    for c in bad:
        top = np.argsort(-M[c])[:3]
        log(f"      类 {c:>3}  召回 {rec[c]:.3f}  → 主要被认成 "
            + ", ".join(f"{t}({M[c, t]})" for t in top if t != c)[:40])

    log("")
    log("=" * 78)
    log("② 混淆最严重的类对（双向误判率之和）")
    log("=" * 78)
    pairs = []
    for a in range(100):
        for b in range(a + 1, 100):
            r = M[a, b] / max(M[a].sum(), 1) + M[b, a] / max(M[b].sum(), 1)
            if r > 0:
                pairs.append((r, a, b))
    pairs.sort(reverse=True)
    for r, a, b in pairs[:10]:
        log(f"    {a:>3} ↔ {b:>3}   互误 {r:.3f}   ({M[a, b]} 张 {a}->{b}, {M[b, a]} 张 {b}->{a})")

    log("")
    log("=" * 78)
    log("③ 最大可分子集：贪心挑「两两不混淆」的类")
    log("=" * 78)
    # 混淆强度矩阵
    conf = np.zeros((100, 100))
    for a in range(100):
        for b in range(100):
            if a != b:
                conf[a, b] = M[a, b] / max(M[a].sum(), 1) + M[b, a] / max(M[b].sum(), 1)
    order = np.argsort(-rec)          # 先放最容易的类
    chosen = []
    for c in order:
        if all(conf[c, o] < 0.10 for o in chosen):
            chosen.append(c)
    log(f"    阈值：任一类对的双向互误 < 0.10")
    log(f"    可容纳类数：{len(chosen)} / 100")
    log(f"    这批类在测试集上的整体召回："
        f"{np.mean([rec[c] for c in chosen]):.4f}")

    out = dict(mean_recall=float(rec.mean()), recall=rec.tolist(),
               n_ge_09=int((rec >= 0.9).sum()), n_ge_08=int((rec >= 0.8).sum()),
               n_ge_05=int((rec >= 0.5).sum()),
               top_confusions=[dict(a=int(a), b=int(b), rate=float(r)) for r, a, b in pairs[:10]],
               separable_subset=int(len(chosen)))
    with open(os.path.join(ROOT, "results", "results_dinov2_capacity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
