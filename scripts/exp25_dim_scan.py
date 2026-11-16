"""exp25: 类多时坐标该取多少维？—— CIFAR-100 维度扫描

exp24 发现：CIFAR-100 上 PCA-16 比全维掉 18.5 点（CIFAR-10 只掉 0.9 点）。
说明"16 维够用"是 CIFAR-10 的结论，不能外推到 100 类。

扫描 d = 8/16/32/64/128/256/384，报：
  ① 原型级 1-NN（路由实际用的量）
  ② 按类去重 top-3 候选召回
  ③ 坐标字节数
并给出"达到全维水平 99% 所需的最小 d"。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def pca_basis(H, kmax=384):
    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    return mu[0], V[:, ::-1][:, :kmax].astype(np.float32), w[::-1]


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    y = d["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:3000]

    mu, V, ev = pca_basis(H)
    # 方差解释比
    cum = np.cumsum(ev) / ev.sum()
    for k in [16, 32, 64, 128, 256, 384]:
        log(f"  前 {k:>3} 维解释方差 {cum[k-1]:.4f}")

    protos_full = np.stack([H[y == c].mean(0) for c in range(100)]).astype(np.float32)
    cls_ids = np.arange(100)

    log("\n" + "=" * 74)
    log(f"  {'d':>4}{'字节/原型':>10}{'原型1-NN':>10}{'单样本1-NN':>11}"
        f"{'召回@3':>9}{'召回@10':>9}{'召回@20':>9}")
    log("-" * 74)
    rows = []
    for k in [8, 16, 32, 64, 128, 256, 384]:
        W = V[:, :k]
        Zp = (protos_full - mu) @ W
        Zq = (T - mu) @ W
        D = dist(Zq, Zp)
        order = np.argsort(D, 1)
        p1 = float(np.mean(cls_ids[order[:, 0]] == yT))

        # 单样本 1-NN（只在 k<=64 时算，避免重复大矩阵乘太慢）
        if k <= 64:
            ZH = (H - mu) @ W
            idx = np.empty(len(T), np.int64)
            gn = (ZH ** 2).sum(1)
            for i in range(0, len(T), 256):
                a = (Zq[i:i + 256] ** 2).sum(1)[:, None] + gn[None, :] - 2 * Zq[i:i + 256] @ ZH.T
                idx[i:i + 256] = np.argmin(a, 1)
            s1 = float(np.mean(y[idx] == yT))
        else:
            s1 = float("nan")

        rec = {}
        for kk in [3, 10, 20]:
            hit = 0
            for i in range(len(yT)):
                seen = []
                for j in order[i]:
                    c = int(cls_ids[j])
                    if c not in seen:
                        seen.append(c)
                    if len(seen) == kk:
                        break
                if int(yT[i]) in seen:
                    hit += 1
            rec[kk] = hit / len(yT)

        log(f"  {k:>4}{k*4:>10}{p1:>10.4f}{s1:>11.4f}"
            f"{rec[3]:>9.4f}{rec[10]:>9.4f}{rec[20]:>9.4f}")
        rows.append(dict(d=k, bytes=k * 4, proto1=p1, sample1=s1,
                         r3=rec[3], r10=rec[10], r20=rec[20],
                         var_explained=float(cum[k - 1])))

    full = [r for r in rows if r["d"] == 384][0]
    ok = [r for r in rows if r["proto1"] >= 0.99 * full["proto1"]]
    log(f"  达到全维原型 1-NN 的 99% 所需最小 d："
        f"{ok[0]['d'] if ok else '>384'} 维"
        f"（{ok[0]['proto1']:.4f} vs 全维 {full['proto1']:.4f}）" if ok else "  >384")
    ok95 = [r for r in rows if r["proto1"] >= 0.95 * full["proto1"]]
    if ok95:
        log(f"  达到全维 95% 所需最小 d：{ok95[0]['d']} 维（{ok95[0]['proto1']:.4f}）")

    with open(os.path.join(ROOT, "results", "results_dim_scan.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
