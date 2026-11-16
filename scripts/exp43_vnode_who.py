"""exp43: 回答"vnode 多了、同类都在空间里集中，为什么反而找不到"

exp34 的缺陷（本实验要修正的）：
  ① 它用"合成撒点"派生 vnode，同类 vnode 的位置是随机噪声，不是真实数据形状；
  ② 它只报 hit_root（终点类 == 全局最近 vnode 的类），没有报
     "终点类 == 真实类"（真正关心的命中率），也没报全局扫描能到的上界；
  ③ 它把"副本增长"和"类数增长"混在同一张表里，导致归因错误。

本实验三条线：
  A 副本增长（K 固定 100，每类 vnode 数 per = 1/3/10/30/100，用真实样本）
  B 类数增长（per 固定 3，K = 10/20/50/100，从 CIFAR-100 抽类）
  C 集中度实测（同类 vnode 两两距 vs 异类 vnode 两两距）

指标定义（关键）：
  oracle  = 全局最近 vnode 的类 == 真实类       ← 路由能达到的上界
  hit     = 贪婪终点 vnode 的类 == 真实类       ← 真正要的命中率
  exact   = 贪婪终点 == 全局最近 vnode          ← 要不要"精确"到那一个
  gap     = oracle - hit                        ← 路由自身走错造成的损失
"""
import os
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DIM = 64


def log(*a):
    print(*a, flush=True)


def sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T, 0)


def knn_adj(Z, cls_of, k=10, block=2000, per_cls_cap=2, want_mult=6):
    """kNN 图（对称化），邻居按类去重：每类最多 per_cls_cap 个。

    返回 list[set]；同时返回跨类边比例。
    """
    M = len(Z)
    adj = [set() for _ in range(M)]
    want = min(M - 1, max(k * want_mult, 48))
    for s in range(0, M, block):
        e = min(s + block, M)
        D = sqdist(Z[s:e], Z)
        for r in range(e - s):
            D[r, s + r] = np.inf
        part = np.argpartition(D, min(want, M - 2), axis=1)[:, :want]
        for r in range(e - s):
            u = s + r
            cnt, sel = {}, []
            cu = cls_of[u]
            for j in part[r]:
                cj = int(cls_of[j])
                if cnt.get(cj, 0) < (per_cls_cap if cj == cu else per_cls_cap):
                    cnt[cj] = cnt.get(cj, 0) + 1
                    sel.append(int(j))
                    if len(sel) >= k:
                        break
            for j in sel:
                adj[u].add(j)
                adj[j].add(u)
        del D
    cross = 0
    tot = 0
    for u in range(M):
        for v in adj[u]:
            tot += 1
            if cls_of[u] != cls_of[v]:
                cross += 1
    return adj, (cross / tot if tot else 0.0)


def greedy(Z, adj, q, start, max_hop=300):
    cur, h = start, 0
    dcur = float(((Z[cur] - q) ** 2).sum())
    while h < max_hop:
        cand = list(adj[cur])
        if not cand:
            break
        cidx = np.fromiter(cand, dtype=np.int32, count=len(cand))
        dd = ((Z[cidx] - q) ** 2).sum(1)
        j = int(np.argmin(dd))
        if dd[j] < dcur - 1e-12:
            cur = cidx[j]
            dcur = float(dd[j])
            h += 1
        else:
            break
    return cur, h


def conc(Z, cls_of):
    """同类 vnode 两两距离 / 异类 vnode 两两距离（采样，避免 O(M^2)）"""
    M = len(Z)
    uniq = np.unique(cls_of)
    intra, inter = [], []
    rng = np.random.RandomState(1)
    per = max(200, 4000 // max(1, len(uniq)))
    for c in uniq[:200]:
        idx = np.where(cls_of == c)[0]
        if len(idx) < 2:
            continue
        a = rng.choice(idx, size=min(per, len(idx)), replace=False)
        b = rng.choice(idx, size=min(per, len(idx)), replace=False)
        d = np.sqrt(sqdist(Z[a], Z[b]))
        intra += list(d[np.triu_indices(len(a), 1)])
    for c in uniq[:100]:
        idx = np.where(cls_of == c)[0]
        oth = np.where(cls_of != c)[0]
        a = rng.choice(idx, size=min(per, len(idx)), replace=False)
        b = rng.choice(oth, size=min(per, len(oth)), replace=False)
        inter += list(np.sqrt(sqdist(Z[a], Z[b])).ravel())
    return float(np.mean(intra)) if intra else 0.0, float(np.mean(inter)) if inter else 0.0


def run_one(Zq, yT, Z, cls_of, k=10, per_cls_cap=2, n_query=500, n_start=1, seed=0):
    """注意：查询子集由 seed 决定；起点随机数用 rng 之后的状态。
    调用方须对同一组对比传同一个 seed（这样两个 n_start 用的是同一批查询）。"""
    rng = np.random.RandomState(seed)
    M = len(Z)
    qi = rng.choice(len(Zq), size=min(n_query, len(Zq)), replace=False)
    Q = Zq[qi]
    yq = yT[qi]

    # 全局最近（分块精确）
    bd = np.full(len(Q), np.inf, np.float32)
    bi = np.zeros(len(Q), np.int32)
    for s in range(0, M, 4000):
        e = min(s + 4000, M)
        D = sqdist(Q, Z[s:e])
        j = D.argmin(1)
        v = D[np.arange(len(Q)), j]
        up = v < bd
        bd[up] = v[up]
        bi[up] = s + j[up]

    oracle = float((cls_of[bi] == yq).mean())

    adj, cross = knn_adj(Z, cls_of, k=k, per_cls_cap=per_cls_cap)

    hit = 0
    exact = 0
    hops = []
    for t in range(len(Q)):
        best_cur, best_h = None, 0
        best_d = np.inf
        for _ in range(n_start):
            cur, h = greedy(Z, adj, Q[t], rng.randint(M))
            d = float(((Z[cur] - Q[t]) ** 2).sum())
            if d < best_d:
                best_d, best_cur, best_h = d, cur, h
        hops.append(best_h)
        if cls_of[best_cur] == yq[t]:
            hit += 1
        if best_cur == bi[t]:
            exact += 1
    n = len(Q)
    deg = float(np.mean([len(a) for a in adj]))
    return dict(oracle=oracle, hit=hit / n, exact=exact / n, gap=oracle - hit / n,
                hops=float(np.mean(hops)), hop_p95=float(np.percentile(hops, 95)),
                deg=deg, cross_edge=cross)


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]
    H = H[:len(y)]
    T = T[:len(yT)]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -DIM:].astype(np.float32)
    Zq_all = (T - mu[0]) @ W
    log(f"特征 {H.shape} → 坐标 d={DIM}；训练 {len(y)}，测试 {len(yT)}")

    # 每类真实样本的坐标（按类分桶）
    Ztr_all = (H - mu[0]) @ W
    by_cls = {c: Ztr_all[y == c] for c in range(100)}
    rng = np.random.RandomState(0)

    out = {"A_replica": [], "B_classes": [], "C_conc": []}

    # ---------------- A: 副本增长，K 固定 100，vnode = 真实样本 ----------------
    log("")
    log("=== A 副本增长（K=100 固定，vnode 取真实训练样本）===")
    log(f"{'per':>5}{'M':>7}{'上界oracle':>11}{'1起点hit':>10}{'3起点hit':>10}"
        f"{'exact':>8}{'gap':>7}{'跳数':>6}{'度':>6}{'跨类边':>8}{'类内距':>8}{'类间距':>8}{'内/间':>8}")
    per_list = [0, 1, 3, 10, 30, 100]   # per=0 表示"每类 1 个类均值原型"（去噪对照）
    for per in per_list:
        if per == 0:
            Z = np.stack([by_cls[c].mean(0) for c in range(100)]).astype(np.float32)
        else:
            Z = np.concatenate([by_cls[c][:per] for c in range(100)]).astype(np.float32)
        cls_of = np.repeat(np.arange(100), max(per, 1)).astype(np.int32)
        M = len(Z)
        k = int(round(max(8, 6 + 4 * math.log10(M / 100))))
        r1 = run_one(Zq_all, yT, Z, cls_of, k=k, n_start=1, seed=1)
        r3 = run_one(Zq_all, yT, Z, cls_of, k=k, n_start=3, seed=1)
        intra, inter = conc(Z, cls_of)
        log(f"{per:>5}{M:>7}{r1['oracle']:>11.4f}{r1['hit']:>10.4f}{r3['hit']:>10.4f}"
            f"{r1['exact']:>8.4f}{r1['gap']:>7.4f}{r1['hops']:>6.2f}{r1['deg']:>6.1f}"
            f"{r1['cross_edge']:>8.3f}{intra:>8.2f}{inter:>8.2f}{intra/inter:>8.3f}")
        out["A_replica"].append(dict(per=per, M=M, k=k, intra=intra, inter=inter,
                                     **{f"h1_{kk}": v for kk, v in r1.items()},
                                     **{f"h3_{kk}": v for kk, v in r3.items()}))
        out["C_conc"].append(dict(per=per, M=M, intra=intra, inter=inter, ratio=intra / inter))

    # ---------------- B: 类数增长，per 固定 3 ----------------
    log("")
    log("=== B 类数增长（per=3 固定，从 CIFAR-100 随机抽 K 类，3 次重复）===")
    log(f"{'K':>5}{'M':>7}{'上界oracle':>11}{'1起点hit':>10}{'3起点hit':>10}"
        f"{'exact':>8}{'gap':>7}{'跳数':>6}{'度':>6}{'跨类边':>8}")
    for K in [10, 20, 50, 100]:
        acc = []
        for rep in range(3 if K < 100 else 1):
            rr = np.random.RandomState(100 + rep)
            cls_sel = np.sort(rr.choice(100, size=K, replace=False))
            # 关键：只保留属于 cls_sel 的查询，并把标签重映射到 0..K-1
            remap = -np.ones(100, dtype=np.int32)
            remap[cls_sel] = np.arange(K, dtype=np.int32)
            keep = remap[yT] >= 0
            ZqK = Zq_all[keep]
            yqK = remap[yT[keep]]
            Z = np.concatenate([by_cls[c][:3] for c in cls_sel]).astype(np.float32)
            cls_of = np.repeat(np.arange(K), 3).astype(np.int32)
            m = len(Z)
            k = int(round(max(8, 6 + 4 * math.log10(max(m, 2) / 100))))
            r1 = run_one(ZqK, yqK, Z, cls_of, k=k, n_start=1, seed=3)
            r3 = run_one(ZqK, yqK, Z, cls_of, k=k, n_start=3, seed=3)
            acc.append((r1, r3, len(ZqK)))
        avg1 = {kk: float(np.mean([a[0][kk] for a in acc])) for kk in acc[0][0]}
        avg3 = float(np.mean([a[1]["hit"] for a in acc]))
        nq = int(np.mean([a[2] for a in acc]))
        log(f"{K:>5}{len(Z):>7}{avg1['oracle']:>11.4f}{avg1['hit']:>10.4f}{avg3:>10.4f}"
            f"{avg1['exact']:>8.4f}{avg1['gap']:>7.4f}{avg1['hops']:>6.2f}{avg1['deg']:>6.1f}"
            f"{avg1['cross_edge']:>8.3f}  (查询 {nq})")
        out["B_classes"].append(dict(K=K, M=len(Z), k=k, hit3=avg3, n_query=nq,
                                     **{f"h1_{kk}": v for kk, v in avg1.items()}))

    with open(os.path.join(ROOT, "results", "results_vnode_who.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("")
    log("已存 results_vnode_who.json")


if __name__ == "__main__":
    main()
