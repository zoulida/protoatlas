"""exp9: 路由坐标的压缩方案对比 —— PCA 是不是唯一/最好的选择？

背景：vnode 坐标要在 gossip 里互传，384 维 float32 = 1536 B/原型 太重。
      之前只比过"PCA 降维"，这里把 ANN 领域成熟的压缩手段一起比。

对比 5 族：
  A. PCA-k float32        (现状)
  B. PCA-k int8 标量量化
  C. ITQ (PCA+旋转+符号)  -> 64/128/256 bit，Hamming 距离
  D. PQ 乘积量化          -> m 段 x 8 bit
  E. 全维 384 基线

指标：1-NN 同类率（贪婪路由的代理）、top3 同类覆盖、与 384 维原空间 top10 重合率、每原型字节数
"""
import os
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def load_labels_aligned():
    """feat_train.npy 的行序 = exp1 里 tr_sel 的顺序。"""
    import sys
    sys.path.insert(0, ROOT)
    import exp1_route as E

    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    node_classes, node_idx = E.make_shard2(ytr, C["n_nodes"], C["n_classes"],
                                           C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    sel = []
    for n, idx in enumerate(node_idx):
        for k in node_classes[n]:
            sub = idx[ytr[idx] == k]
            take = rng.choice(sub, size=min(C["per_class_train"], len(sub)), replace=False)
            sel.append(take)
    return ytr[np.concatenate(sel)].astype(np.int32)


def d2mat(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def metrics(d2, yq, yg, top_raw, k=10):
    order = np.argsort(d2, axis=1)[:, :k]
    nn1 = float(np.mean(yg[order[:, 0]] == yq))
    top3 = float(np.mean([np.mean(yg[order[i, :3]] == yq[i]) for i in range(len(yq))]))
    ov = float(np.mean([len(set(order[i].tolist()) & set(top_raw[i].tolist())) / k
                        for i in range(len(yq))]))
    return nn1, top3, ov


def kmeans_np(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = d2mat(X, C).argmin(1)
        cnt = np.bincount(a, minlength=k)
        sums = np.stack([np.bincount(a, weights=X[:, j], minlength=k)
                         for j in range(X.shape[1])], 1)
        C = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    return C


def itq_fit(Z, bits, iters=50, seed=0):
    """在参照集上拟合正交旋转 R；query 必须复用同一个 R，否则两边编码不可比。"""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(bits, bits).astype(np.float32))
    R = Q
    for _ in range(iters):
        B = np.sign(Z @ R)
        U, S, Vt = np.linalg.svd(B.T @ Z, full_matrices=False)
        R = (U @ Vt).astype(np.float32)
    return R


def itq_apply(Z, R):
    return (Z @ R > 0).astype(np.float32)


def hamming_d2(bq, bg):
    """b in {0,1} -> 返回"相似度代价"（越小越近）。"""
    return bq @ (1.0 - bg).T + (1.0 - bq) @ bg.T


def pq_d2(Q, G, Cq, Cg, m):
    """非对称距离：query 不量化，gallery 用量化码。"""
    dim = Q.shape[1]
    step = dim // m
    d2 = np.zeros((len(Q), len(G)), dtype=np.float32)
    for s in range(m):
        sl = slice(s * step, (s + 1) * step)
        cs = Cq[s]
        tab = ((Q[:, sl] ** 2).sum(1)[:, None]
               + (cs ** 2).sum(1)[None, :]
               - 2.0 * (Q[:, sl] @ cs.T))
        d2 += tab[:, Cg[:, s]]
    return np.maximum(d2, 0)


def main():
    t0 = time.time()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    y = load_labels_aligned()
    assert len(y) == len(H), f"标签 {len(y)} 与特征 {len(H)} 不匹配"

    rng = np.random.RandomState(0)
    n_query, n_gallery = 1000, 5000
    idx = rng.permutation(len(H))
    qi, gi = idx[:n_query], idx[n_query:n_query + n_gallery]
    Hq, Hg = H[qi], H[gi]
    yq, yg = y[qi], y[gi]

    mu = H.mean(0, keepdims=True)
    Hc = H - mu
    cov = (Hc.T @ Hc) / (len(Hc) - 1)
    w, V = np.linalg.eigh(cov.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)
    ev = w[::-1]
    log(f"主成分解释方差: 前8 {ev[:8].sum()/ev.sum()*100:.1f}%  "
        f"前16 {ev[:16].sum()/ev.sum()*100:.1f}%  前64 {ev[:64].sum()/ev.sum()*100:.1f}%")

    Zq_all = (Hq - mu) @ V
    Zg_all = (Hg - mu) @ V

    rows = []

    def add(name, d2, nbytes, note=""):
        nn1, top3, ov = metrics(d2, yq, yg, top_raw)
        rows.append(dict(method=name, bytes=nbytes, nn1=nn1, top3=top3, top10_overlap=ov, note=note))
        log(f"{name:22s} {nbytes:6d} B  nn1={nn1:.4f}  top3={top3:.4f}  与原空间top10重合={ov:.3f}")

    top_raw = np.argsort(d2mat(Zq_all, Zg_all), axis=1)[:, :10]

    add("全维 384 f32", d2mat(Zq_all, Zg_all), 384 * 4, "基线，无压缩")

    for k in [8, 16, 32, 64, 128]:
        add(f"PCA-{k} f32", d2mat(Zq_all[:, :k], Zg_all[:, :k]), k * 4)

    for k, qb in [(16, 16), (32, 16), (16, 8), (64, 8)]:
        A = Zg_all[:, :k]
        lo = np.percentile(A, 1, axis=0)
        hi = np.percentile(A, 99, axis=0)
        mx = float(2 ** (qb - 1) - 1)
        sc = mx / np.maximum(hi - lo, 1e-8)
        qq = np.clip(np.round((Zq_all[:, :k] - lo) * sc), -mx, mx).astype(np.float32)
        gg = np.clip(np.round((A - lo) * sc), -mx, mx).astype(np.float32)
        add(f"PCA-{k} int{qb}", d2mat(qq, gg), k * qb // 8, f"每维 {qb} 位，1%截尾")

    for bits in [64, 128, 256]:
        R = itq_fit(Zg_all[:, :bits], bits)
        bq = itq_apply(Zq_all[:, :bits], R)
        bg = itq_apply(Zg_all[:, :bits], R)
        add(f"ITQ-{bits}bit", hamming_d2(bq, bg), bits // 8, "Hamming 距离")

    for m in [8, 16, 48]:
        step = 384 // m
        cents, CG = [], []
        for s in range(m):
            sl = slice(s * step, (s + 1) * step)
            C = kmeans_np(Zg_all[:, sl], 256, iters=20, seed=s)
            cents.append(C)
            CG.append(d2mat(Zg_all[:, sl], C).argmin(1).astype(np.int32))
        CG = np.stack(CG, 1)
        d2 = np.zeros((n_query, n_gallery), dtype=np.float32)
        for s in range(m):
            sl = slice(s * step, (s + 1) * step)
            tab = ((Zq_all[:, sl] ** 2).sum(1)[:, None]
                   + (cents[s] ** 2).sum(1)[None, :]
                   - 2.0 * (Zq_all[:, sl] @ cents[s].T))
            d2 += tab[:, CG[:, s]]
        add(f"PQ-{m}x8bit", np.maximum(d2, 0), m, f"{step} 维/段")

    with open(os.path.join(ROOT, "results", "results_compress.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    log("\n--- 按字节数排序（同字节数下取 nn1 最高的方案）---")
    best = {}
    for r in rows:
        b = r["bytes"]
        if b not in best or r["nn1"] > best[b]["nn1"]:
            best[b] = r
    for b in sorted(best):
        r = best[b]
        log(f"{b:6d} B  ->  {r['method']:18s} nn1={r['nn1']:.4f}")
    log(f"\n完成 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
