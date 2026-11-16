"""exp28e: 表 5-2（按类去重）的判别投影口径复核

表 5-2 现给「全维 384 / 主成分 16 维」两列，vnode 为节点级原型
（20 节点 × 2 类 = 40 个 vnode，每类 4 个副本）。本脚本按同一 vnode 构造
补算判别投影 d=K-1=9 一列，并复算另两列以核对旧读数。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fit_lda(H, y, d, lam=1e-3):
    if os.environ.get("NO_PROJ"):
        # 去掉判别投影：坐标即原特征，维数等于特征维数
        return (np.zeros(H.shape[1], np.float32),
                np.eye(H.shape[1], dtype=np.float32), int(H.shape[1]))
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
    diff3 = float(np.mean([len(set(cv[o[i, :3]].tolist())) for i in range(nq)]))

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
    print(f"{tag:12s} 命中@1 {hit1:.4f}  未去重top3不同类 {diff3:.2f}  "
          f"去重top3 {r3:.4f}  去重top5 {r5:.4f}", flush=True)
    return dict(dim=0, hit1=hit1, diff3=diff3, r3=r3, r5=r5)


def main():
    from exp13_dedup_reject import build
    H, T, P, pcls, pnode, y_tr, y_te, tr_node, nc, C = build()
    print(f"vnode 数 {len(P)}，类数 {len(set(pcls.tolist()))}，查询 {len(T)}")

    mu0 = H.mean(0, keepdims=True).astype(np.float32)
    Cc = (H - mu0).T @ (H - mu0) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W16 = V[:, ::-1][:, :16].astype(np.float32)
    mu_lda, W_lda, d_eff = fit_lda(H, y_tr, 9)

    out = {}
    out["lda"] = dict(stats((T - mu_lda) @ W_lda, y_te, (P - mu_lda) @ W_lda, pcls,
                            f"lda d={d_eff}"), d=int(d_eff))
    out["raw384"] = dict(stats(T - mu0, y_te, P - mu0, pcls, "raw 384"), d=384)
    out["pca16"] = dict(stats((T - mu0) @ W16, y_te, (P - mu0) @ W16, pcls, "pca 16"), d=16)

    with open(os.path.join(ROOT, "results", "results_dedup_lda_vnode.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("写入 results_dedup_lda_vnode.json")


if __name__ == "__main__":
    main()
