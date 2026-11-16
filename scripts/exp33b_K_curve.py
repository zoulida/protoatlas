"""exp33b: 纯 K 扫描 —— 同一数据集(CIFAR-100 细标签)内取前 K 类，K=5..100

exp33 的教训：拿 CIFAR-100 的 20 个 coarse 类当"更简单的 K=20"是错的，
粗类是多模态合并（"水生哺乳动物"= 5 个细类），类内散布反而更大。
必须固定数据集与粒度，只变类数。

做法：从 100 个细类里随机抽 K 个（不放回），重复 R 次取平均。
  K = 5 / 10 / 20 / 50 / 100
  每类训练 500，每类测试最多 30
  d = 8 .. 384

输出：
  ① 召回@1/@3/@10 随 (K,d) 的二维表
  ② d*(K)：达到 @3 = 0.95 的最小维度
  ③ 拟合 d* ∝ K^α，给"每涨 10 倍类数需要涨多少倍维度"
  ④ 可分性指标：最近邻类中心距 / 类内平均半径（随 K 衰减，解释维度需求）
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def rec(D, yq, cls_p, ks=(1, 3, 10)):
    order = np.argsort(D, 1)
    out = {}
    for k in ks:
        got = cls_p[order[:, :k]]
        out[k] = float(np.mean((got == yq[:, None]).any(1)))
    return out


def main():
    import pandas as pd
    from collections import defaultdict

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]

    dims = [8, 16, 32, 64, 128, 192, 256, 384]
    Ks = [5, 10, 20, 50, 100]
    R = 3
    rng = np.random.RandomState(7)

    # 每类索引
    idx_tr = {c: np.where(y == c)[0][:500] for c in range(100)}
    idx_te = {c: np.where(yT == c)[0] for c in range(100)}

    agg = defaultdict(lambda: defaultdict(list))   # K -> d -> [r3,...]
    agg1 = defaultdict(lambda: defaultdict(list))
    agg10 = defaultdict(lambda: defaultdict(list))
    sep = defaultdict(list)

    log(f"  {'K':>5}{'rep':>5}{'训练':>8}{'测试':>7}{'类内R':>8}{'类间D':>8}{'D/R':>7}"
        f"{'@1(384)':>9}{'@3(384)':>9}")
    log("-" * 84)

    for K in Ks:
        reps = 1 if K == 100 else R
        for rep in range(reps):
            cls = np.sort(rng.choice(100, size=K, replace=False)) if K < 100 else np.arange(100)
            rows_tr = np.concatenate([idx_tr[c] for c in cls])
            rows_te = np.concatenate([idx_te[c][:30] for c in cls])
            Hs, ys = H[rows_tr], y[rows_tr]
            Ts, yTs = T[rows_te], yT[rows_te]

            mu = Hs.mean(0, keepdims=True)
            Cc = (Hs - mu).T @ (Hs - mu) / (len(Hs) - 1)
            w, V = np.linalg.eigh(Cc.astype(np.float64))
            V = V[:, ::-1].astype(np.float32)

            protos = np.stack([Hs[ys == c].mean(0) for c in cls]).astype(np.float32)
            # 可分性
            r_in = float(np.mean([np.linalg.norm(Hs[i] - protos[np.searchsorted(cls, ys[i])])
                                  for i in range(0, len(Hs), 7)]))
            dp = dist(protos, protos)
            np.fill_diagonal(dp, np.inf)
            d_out = float(dp.min(1).mean())
            sep[K].append(d_out / r_in)

            for dim in dims:
                W = V[:, :dim]
                Zp = (protos - mu) @ W
                Zq = (Ts - mu) @ W
                D = dist(Zq, Zp)
                r = rec(D, yTs, cls)
                agg[K][dim].append(r[3])
                agg1[K][dim].append(r[1])
                agg10[K][dim].append(r[10])
                if dim == 384:
                    last = (r[1], r[3])
            log(f"  {K:>5}{rep:>5}{len(Hs):>8}{len(Ts):>7}{r_in:>8.2f}{d_out:>8.2f}"
                f"{d_out/r_in:>7.3f}{last[0]:>9.4f}{last[1]:>9.4f}")

    log("")
    log("=" * 84)
    log("召回@3 随维度变化（K 次重复平均）")
    log("=" * 84)
    head = "  " + f"{'d':>5}" + "".join(f"{('K='+str(K)):>10}" for K in Ks)
    log(head)
    log("  " + "-" * (len(head) - 2))
    table = {}
    for dim in dims:
        vals = [float(np.mean(agg[K][dim])) for K in Ks]
        table[dim] = vals
        log(f"  {dim:>5}" + "".join(f"{v:>10.4f}" for v in vals))

    log("")
    log("  d*(K)：达到 召回@3 = 0.95 的最小维度")
    dstar = {}
    for K in Ks:
        hit = [d for d in dims if np.mean(agg[K][d]) >= 0.95]
        dstar[K] = hit[0] if hit else None
        v384 = float(np.mean(agg[K][384]))
        log(f"    K={K:<4} d* = {str(hit[0]) if hit else '>384':>5}    (全维 @3 = {v384:.4f})")

    ok = [(K, dstar[K]) for K in Ks if dstar[K]]
    if len(ok) >= 2:
        x = np.log([k for k, _ in ok])
        yv = np.log([d for _, d in ok])
        alpha = float(np.polyfit(x, yv, 1)[0])
    else:
        alpha = float("nan")
    log(f"    拟合 d* ∝ K^{alpha:.2f}  →  K 涨 10 倍，d* 涨 {10**alpha:.1f} 倍")

    log("")
    log("  可分性 D/R（最近邻类中心距 / 类内平均半径，全维）：")
    for K in Ks:
        log(f"    K={K:<4} D/R = {np.mean(sep[K]):.3f}")

    out = dict(ks=Ks, dims=dims, table=table, dstar={str(k): v for k, v in dstar.items()},
               alpha=alpha, sep={str(k): float(np.mean(v)) for k, v in sep.items()},
               r1={str(K): {str(d): float(np.mean(agg1[K][d])) for d in dims} for K in Ks},
               r10={str(K): {str(d): float(np.mean(agg10[K][d])) for d in dims} for K in Ks})
    with open(os.path.join(ROOT, "results", "results_K_curve.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
