"""exp28b: 分布式损失的补救——按判别投影（LDA，d=K-1）重做 exp28

背景：exp28 的 results_greedy_recover.json 是在 d=64 主成分投影上跑的，
而正文主张坐标维度由类数决定（第 4.2.3 节），100 类时 d=99 且为判别投影。
本节实验因此须按同一投影重跑，另保留 PCA 64 维一档作为对照。

流程与 exp28 完全一致，只换投影矩阵：
  D  1 起点 + 1 跳扩散
  E  1 起点 + 2 跳扩散
  F  3 起点并行 + 1 跳扩散
  G  3 起点并行 + 2 跳扩散
  H  3 起点并行 + 2 跳 + top-5 输出
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


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:3000]

    K_CLS, K, LOCAL = 100, 8, 8
    cls_ids = np.arange(K_CLS)

    # ---------------- 两种投影
    mu0 = H.mean(0, keepdims=True)
    Cc = (H - mu0).T @ (H - mu0) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)
    W_pca = V[:, :64]
    mu_lda, W_lda, d_lda = fit_lda(H, y, K_CLS - 1)

    protos = np.stack([H[y == c].mean(0) for c in range(K_CLS)]).astype(np.float32)
    rng = np.random.RandomState(0)

    out = {}
    for tag, mu, W in (("lda", mu_lda, W_lda), ("pca64", mu0.astype(np.float32), W_pca)):
        Zv = (protos - mu) @ W
        Zq = (T - mu) @ W
        Dg = dist(Zq, Zv)
        Dv = dist(Zv, Zv)
        np.fill_diagonal(Dv, np.inf)
        nbr = np.argsort(Dv, 1)[:, :K]
        adj = [set(nbr[i]) for i in range(K_CLS)]
        for i in range(K_CLS):
            for j in nbr[i]:
                adj[j].add(i)
        adj = [sorted(a) for a in adj]
        glob_order = np.argsort(Dg, 1)

        def global_recall(kk):
            hit = 0
            for i in range(len(yT)):
                seen = []
                for j in glob_order[i]:
                    c = int(cls_ids[j])
                    if c not in seen:
                        seen.append(c)
                    if len(seen) == kk:
                        break
                if int(yT[i]) in seen:
                    hit += 1
            return hit / len(yT)

        def greedy(q, start):
            cur, h = start, 0
            while h < 30:
                cand = adj[cur]
                dd = dist(q[None, :], Zv[cand])[0]
                b = cand[int(np.argmin(dd))]
                if dd.min() < dist(q[None, :], Zv[cur:cur + 1])[0]:
                    cur, h = b, h + 1
                else:
                    break
            return cur, h

        def pool_of(cur, radius):
            s = {cur}
            frontier = {cur}
            for _ in range(radius):
                nxt = set()
                for u in frontier:
                    nxt |= set(adj[u])
                frontier = nxt - s
                s |= nxt
            return sorted(s)

        if os.environ.get("GRID"):
            variants = [(f"S{ns}H{r}", ns, r, 3)
                        for ns in (1, 2, 3, 5) for r in (1, 2)]
        else:
            variants = [
            ("D 1起点 + 1跳 + top3", 1, 1, 3),
            ("E 1起点 + 2跳 + top3", 1, 2, 3),
            ("F 3起点 + 1跳 + top3", 3, 1, 3),
            ("G 3起点 + 2跳 + top3", 3, 2, 3),
            ("H 3起点 + 2跳 + top5", 3, 2, 5),
        ]
        rng = np.random.RandomState(0)   # 两档用同一组起点，便于对照
        res = {"proj": tag, "d": int(W.shape[1]),
               "global_r3": global_recall(3), "global_r5": global_recall(5)}
        # 纯贪婪（本地择优起点、不扩散）停在局部极小的比例
        near_global = np.argmin(Dg, 1)
        rng2 = np.random.RandomState(0)
        stuck = 0
        for i in range(len(Zq)):
            q = Zq[i]
            loc = rng2.choice(K_CLS, LOCAL, replace=False)
            s = int(loc[int(np.argmin(dist(q[None, :], Zv[loc])[0]))])
            cur, _ = greedy(q, s)
            stuck += int(cur != int(near_global[i]))
        res["stuck_frac"] = stuck / len(Zq)
        log(f"[{tag}] d={W.shape[1]}  全局上界 @3 {res['global_r3']:.4f} "
            f"@5 {res['global_r5']:.4f}  纯贪婪未达全局最近原型 {res['stuck_frac']:.4f}")
        log(f"  {'方案':<26}{'召回@3':>9}{'召回@5':>9}{'询问vnode':>11}{'跳数':>8}")
        for label, ns, radius, kout in variants:
            # 每个方案用自己的随机流，种子只由 (ns, radius) 决定，
            # 结果因而与 variants 的顺序无关，重跑逐位相同
            rng_v = np.random.RandomState(10007 * ns + 97 * radius + 3)
            r3 = r5 = 0
            qn, hp = [], []
            broute, bprobe = [], []
            msg = 4 * int(W.shape[1]) + 8      # 一个报文的字节数：坐标 + vnode 标识
            for i in range(len(Zq)):
                q = Zq[i]
                loc = rng_v.choice(K_CLS, LOCAL, replace=False)
                starts = [int(loc[j]) for j in np.argsort(dist(q[None, :], Zv[loc])[0])[:ns]]
                pool, hsum = set(), 0
                for s in starts:
                    cur, h = greedy(q, s)
                    hsum += h
                    pool |= set(pool_of(cur, radius))
                pool = sorted(pool)
                qn.append(len(pool))
                hp.append(hsum / ns)
                broute.append(hsum * msg)
                bprobe.append(kout * msg)
                dd = dist(q[None, :], Zv[pool])[0]
                o = np.argsort(dd)
                seen = []
                for j in o:
                    c = int(cls_ids[pool[j]])
                    if c not in seen:
                        seen.append(c)
                    if len(seen) >= 5:
                        break
                r3 += int(int(yT[i]) in seen[:3])
                r5 += int(int(yT[i]) in seen[:5])
            n = len(Zq)
            log(f"  {label:<26}{r3/n:>9.4f}{r5/n:>9.4f}{np.mean(qn):>11.1f}{np.mean(hp):>8.2f}"
                f"{np.mean(broute):>10.0f}{np.mean(bprobe):>10.0f}")
            res[label] = dict(r3=r3 / n, r5=r5 / n,
                              asked=float(np.mean(qn)), hops=float(np.mean(hp)),
                              msg_b=msg,
                              route_b=float(np.mean(broute)),
                              probe_b=float(np.mean(bprobe)),
                              feat_b=1536,
                              total_b=float(np.mean(broute) + np.mean(bprobe) + 1536))
        out[tag] = res

    fn = "results_greedy_grid_lda.json" if os.environ.get("GRID") else "results_greedy_recover_lda.json"
    with open(os.path.join(ROOT, fn), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("\n写入 " + fn)


if __name__ == "__main__":
    main()
