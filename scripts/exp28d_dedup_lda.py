"""exp28d: 按类去重的口径复核（CIFAR-10，判别投影 d=K-1）

正文表 5-2 现给的是「全维 384 / 主成分 16 维」两列，都不是本文口径。
本脚本补算判别投影下的同一组四个指标，并复算全维一列以核对旧读数。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def fit_lda(H, y, d, lam=1e-3):
    mu = H.mean(0).astype(np.float64)
    X = H.astype(np.float64) - mu
    cls = np.unique(y)
    K, D = len(cls), H.shape[1]
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in cls:
        m = y == c
        Xc = X[m]
        mc = Xc.mean(0)
        Sw += Xc.T @ Xc - len(Xc) * np.outer(mc, mc)
        Sb += len(Xc) * np.outer(mc, mc)
    Sw /= max(1, len(X) - K)
    Sb /= max(1, K - 1)
    tr = np.trace(Sw) / D
    Sw_r = Sw + lam * tr * np.eye(D)
    L = np.linalg.cholesky(Sw_r)
    Li = np.linalg.inv(L)
    A = Li @ Sb @ Li.T
    A = (A + A.T) / 2
    w, U = np.linalg.eigh(A)
    d_eff = min(d, K - 1)
    idx = np.argsort(w)[::-1][:d_eff]
    W = Li.T @ U[:, idx]
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return mu.astype(np.float32), W.astype(np.float32), d_eff


def d2(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


def stats(Zq, yq, Zv, cv, tag):
    D = d2(Zq, Zv)
    o = np.argsort(D, 1)
    nq = len(yq)
    hit1 = float((cv[o[:, 0]] == yq).mean())
    # 未去重 top-3 中不同类数
    diff3 = float(np.mean([len(set(cv[o[i, :3]].tolist())) for i in range(nq)]))
    # 去重后 top-k 召回
    def dedup_rec(k):
        hit = 0
        for i in range(nq):
            seen = []
            for j in o[i]:
                c = int(cv[j])
                if c not in seen:
                    seen.append(c)
                if len(seen) == k:
                    break
            hit += int(int(yq[i]) in seen)
        return hit / nq
    r3, r5 = dedup_rec(3), dedup_rec(5)
    print(f"{tag:14s} 命中@1 {hit1:.4f}  未去重top3不同类 {diff3:.2f}  "
          f"去重top3 {r3:.4f}  去重top5 {r5:.4f}", flush=True)
    return dict(hit1=hit1, diff3=diff3, r3=r3, r5=r5)


def main():
    sys.path.insert(0, ROOT)
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr_all, yte_all, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    ytr = ytr_all[tr_sel].astype(np.int32)
    yte = yte_all[te_sel].astype(np.int32)
    n = min(len(T), 3000)
    T, yte = T[:n], yte[:n]

    K = 10
    cls = np.arange(K)
    protos = np.stack([H[ytr == c].mean(0) for c in cls]).astype(np.float32)

    mu_lda, W_lda, d_eff = fit_lda(H, ytr, K - 1)
    mu0 = H.mean(0, keepdims=True).astype(np.float32)

    out = {}
    out["lda"] = dict(d=int(d_eff), **stats((T - mu_lda) @ W_lda, yte,
                                            (protos - mu_lda) @ W_lda, cls, f"lda d={d_eff}"))
    out["raw384"] = dict(d=384, **stats(T - mu0, yte, protos - mu0, cls, "raw 384"))
    # 主成分 16 维（复现旧列）
    Cc = (H - mu0).T @ (H - mu0) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W16 = V[:, ::-1][:, :16].astype(np.float32)
    out["pca16"] = dict(d=16, **stats((T - mu0) @ W16, yte, (protos - mu0) @ W16, cls, "pca 16"))

    with open(os.path.join(ROOT, "results", "results_dedup_lda.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("写入 results_dedup_lda.json")


if __name__ == "__main__":
    main()
