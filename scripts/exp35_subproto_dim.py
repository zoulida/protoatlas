"""exp35: 维度不够时能不能用子原型把 d* 买回来

exp33b 结论：单原型下 d*(K) ∝ K^0.93，K=100 时全维 384 都到不了 @3=0.95。
但那个 0.95 是被"每类一个原型"的表示能力卡住的，不是维度。

本实验检验：每类拆成 m 个子中心（类内 k-means），能否在更低维度达标。
  m = 1 / 2 / 4 / 8
  d = 16 / 32 / 64 / 128 / 256 / 384
指标一律**按类去重**：沿距离升序，同类只留最近一个，再取前 k 个不同类。
（不去重的话子原型会让 top-3 被同类占满，数字虚高——这是 exp13 踩过的坑）
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


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        D = dist(X, C)
        a = D.argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C


def rec_dedup(D, yq, cls_of_proto, ks=(1, 3, 10)):
    """按类去重的召回@k"""
    order = np.argsort(D, 1)
    cls_sorted = cls_of_proto[order]
    out = {}
    for k in ks:
        hit = 0
        for i in range(len(yq)):
            seen, picked = [], []
            for c in cls_sorted[i]:
                if c not in seen:
                    seen.append(c)
                    picked.append(c)
                    if len(picked) >= k:
                        break
            if yq[i] in picked:
                hit += 1
        out[k] = hit / len(yq)
    return out


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)

    dims = [16, 32, 64, 128, 256, 384]
    Ms = [1, 2, 4, 8]
    res = {}
    log(f"  {'m':>3}{'d':>6}{'原型数':>7}{'@1':>8}{'@3':>8}{'@10':>8}")
    log("-" * 44)
    for m in Ms:
        protos, pcls = [], []
        for c in range(100):
            Xc = H[y == c]
            if m == 1:
                cs = Xc.mean(0, keepdims=True)
            else:
                cs = kmeans(Xc, m, seed=c)
            protos.append(cs)
            pcls.append(np.full(len(cs), c))
        P = np.concatenate(protos).astype(np.float32)
        pc = np.concatenate(pcls)
        for d in dims:
            W = V[:, :d]
            Zp = (P - mu) @ W
            Zq = (T - mu) @ W
            r = rec_dedup(dist(Zq, Zp), yT, pc)
            res.setdefault(str(m), {})[str(d)] = r
            log(f"  {m:>3}{d:>6}{len(P):>7}{r[1]:>8.4f}{r[3]:>8.4f}{r[10]:>8.4f}")
        log("")

    log("  d*(m)：达到 @3 = 0.95 的最小维度")
    for m in Ms:
        hit = [d for d in dims if res[str(m)][str(d)][3] >= 0.95]
        log(f"    m={m:<3} d* = {hit[0] if hit else '>384'}")

    with open(os.path.join(ROOT, "results", "results_subproto_dim.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
