"""exp34: vnode 规模能撑多大 —— 类数增长 vs 副本数增长，以及分层长链的作用

问题：极限情况下可参与的节点有多少，节点数上限该设多少？

关键区分（之前一直混着说）：
  vnode 总数 M = Σ(每节点自己的类数 × 每类子原型数)
  M 增长有两个来源，对路由的伤害完全不同：
    (A) 类数 K 增长  → 簇数变多，簇间空白变大 → 跳数真会变长   ← 真瓶颈
    (B) 副本数增长   → 同类在更多节点上有 vnode，簇内更密 → 反而更好找

本实验用真实的 CIFAR-100 类中心（PCA-64 坐标）做种子，层级派生出大 K：
  第 0 层 100 个真中心 → 第 1 层每个派生 10 个（K=1000）→ 第 2 层再派生 10 个（K=10000）
派生尺度逐层递减，模拟 ImageNet "狗 → 120 品种" 的语义层级。

对比两种路由：
  B1  仅 kNN(k=8) 贪婪下降
  B2  kNN + 分层长链（每个节点在各距离环内随机 1 条捷径，Kleinberg 风格）
"""
import os
import sys
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T, 0)


def knn_graph(Z, k=8, block=2000, symmetrize=True, root_of=None, per_class_cap=2):
    """分块算 kNN，返回邻接表 list[set]（对称化）。

    若给 root_of，则邻居按类去重（每类最多 per_class_cap 个）——
    这正是方案 L2 的设计：vnode 副本多了以后，邻居表会被同类副本占满，
    必须按类去重才能保留跨类边。
    """
    M = len(Z)
    adj = [set() for _ in range(M)]
    want = min(M - 1, max(k * 8, 64) if root_of is not None else k)
    for s in range(0, M, block):
        e = min(s + block, M)
        D = sqdist(Z[s:e], Z)
        for r in range(e - s):
            D[r, s + r] = np.inf
        part = np.argpartition(D, min(want, M - 2), axis=1)[:, :want]
        for r in range(e - s):
            u = s + r
            if root_of is None:
                sel = part[r][:k]
            else:
                cnt, sel = {}, []
                for j in part[r]:
                    cj = int(root_of[j])
                    if cnt.get(cj, 0) < per_class_cap:
                        cnt[cj] = cnt.get(cj, 0) + 1
                        sel.append(int(j))
                        if len(sel) >= k:
                            break
                sel = np.array(sel, dtype=np.int32)
            for j in sel:
                j = int(j)
                adj[u].add(j)
                if symmetrize:
                    adj[j].add(u)
        del D
    return adj


def full_dist_block(Z, s, e):
    D = sqdist(Z[s:e], Z)
    rows = np.arange(e - s)
    D[rows, np.arange(s, e)] = np.inf
    return D


def make_levels(K0=100, branch=10, levels=3, seed=0):
    """返回 (Z_seed, layer_of, parent0)，Z_seed 是层0 的 K0 个中心"""
    rng = np.random.RandomState(seed)
    return rng  # 占位，实际由调用方提供真中心


def build_hier_centers(c0, branch, levels, sigma0, rng):
    """从层0 中心逐层派生，返回 (最深一层中心, 每个中心所属的层0根索引)"""
    cur = c0.copy()
    root = np.arange(len(c0), dtype=np.int32)
    sig = sigma0
    for _ in range(levels):
        nxt, nrt = [], []
        for i in range(len(cur)):
            off = rng.randn(branch, c0.shape[1]).astype(np.float32) * sig
            nxt.append(cur[i] + off)
            nrt.append(np.full(branch, root[i], np.int32))
        cur = np.concatenate(nxt)
        root = np.concatenate(nrt)
        sig *= 0.45
    return cur, root


def greedy(Z, adj, q, start, max_hop=500):
    cur, h = start, 0
    dcur = float(((Z[cur] - q) ** 2).sum())
    while h < max_hop:
        cand = list(adj[cur])
        if not cand:
            break
        cidx = np.fromiter(cand, dtype=np.int32, count=len(cand))
        dd = ((Z[cidx] - q) ** 2).sum(1)
        j = int(np.argmin(dd))
        if dd[j] < dcur - 1e-9:
            cur = cidx[j]
            dcur = float(dd[j])
            h += 1
        else:
            break
    return cur, h


def main():
    import pandas as pd
    from scipy.spatial import cKDTree

    dim = 64
    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -dim:].astype(np.float32)
    c0 = np.stack([(H[y == c].mean(0) - mu[0]) @ W for c in range(100)]).astype(np.float32)
    Zq_all = (T - mu[0]) @ W
    # 派生扰动尺度：必须远小于类间距离，否则层级结构被冲掉
    dp0 = np.sqrt(sqdist(c0, c0))
    np.fill_diagonal(dp0, np.inf)
    near = float(dp0.min(1).mean())
    sigma0 = near * 0.18

    rng = np.random.RandomState(0)
    log(f"  坐标 d={dim}，层0 中心 100 个，最近类中心距 {near:.2f}，派生尺度 sigma0={sigma0:.2f}")
    log(f"  {'M':>8}{'叶子簇':>9}{'根簇':>7}{'k':>4}{'路由':>13}{'跳1起点':>8}{'跳3起点':>8}"
        f"{'跳p95':>7}{'=全局最近':>10}{'类正确':>9}{'度':>7}")
    log("-" * 100)

    out = []
    cfgs = [dict(levels=0, branch=0, per=3),      # K=100   M=300
            dict(levels=1, branch=10, per=3),     # K=1000  M=3000
            dict(levels=2, branch=10, per=3),     # K=10000 M=30000
            dict(levels=1, branch=10, per=10)]    # K=1000  M=10000（副本增长）
    for cfg in cfgs:
        lv, br, per = cfg["levels"], cfg["branch"], cfg["per"]
        cs, roots = build_hier_centers(c0, br if lv else 1, lv, sigma0, rng)
        leaf = len(cs)
        eps = sigma0 * 0.06
        Z = (cs[:, None, :] + rng.randn(len(cs), per, dim).astype(np.float32) * eps).reshape(-1, dim)
        root_of = np.repeat(roots, per)
        leaf_of = np.repeat(np.arange(len(cs)), per)
        M = len(Z)
        K0 = len(np.unique(root_of))

        qi = rng.choice(len(Zq_all), size=400, replace=False)
        Zq = Zq_all[qi]

        best_d = np.full(len(Zq), np.inf, np.float32)
        best_i = np.zeros(len(Zq), np.int32)
        for s in range(0, M, 5000):
            e = min(s + 5000, M)
            D = sqdist(Zq, Z[s:e])
            j = D.argmin(1)
            v = D[np.arange(len(Zq)), j]
            upd = v < best_d
            best_d[upd] = v[upd]
            best_i[upd] = s + j[upd]

        kg = int(round(max(8, 8 + 5 * math.log10(M / 300))))
        base_adj = knn_graph(Z, k=kg, block=2000, root_of=leaf_of, per_class_cap=2)
        for mode in ["knn", "knn+longlink"]:
            if mode == "knn":
                adj = base_adj
            else:
                adj = [set(a) for a in base_adj]
                J = 4
                for s in range(0, M, 2000):
                    e = min(s + 2000, M)
                    D = full_dist_block(Z, s, e)
                    for r in range(e - s):
                        row = D[r]
                        u = s + r
                        qs = np.quantile(row[np.isfinite(row)], np.linspace(0, 1, J + 1))
                        for j2 in range(J):
                            m = (row >= qs[j2]) & (row < qs[j2 + 1])
                            cand = np.where(m)[0]
                            if len(cand):
                                v = int(cand[rng.randint(len(cand))])
                                adj[u].add(v)
                                adj[v].add(u)
                    del D

            hops1, hops3, exact, hitroot = [], [], 0, 0
            for t in range(len(Zq)):
                cur, h = greedy(Z, adj, Zq[t], rng.randint(M))
                hops1.append(h)
                if cur == best_i[t]:
                    exact += 1
                if root_of[cur] == root_of[best_i[t]]:
                    hitroot += 1
                # 3 起点取最近
                best_h, best_cur = h, cur
                for _ in range(2):
                    c2, h2 = greedy(Z, adj, Zq[t], rng.randint(M))
                    if ((Z[c2] - Zq[t]) ** 2).sum() < ((Z[best_cur] - Zq[t]) ** 2).sum():
                        best_cur, best_h = c2, h2
                hops3.append(best_h)
            deg = float(np.mean([len(a) for a in adj]))
            n = len(Zq)
            log(f"  {M:>8}{leaf:>9}{K0:>7}{kg:>4}{mode:>13}{np.mean(hops1):>7.2f}{np.mean(hops3):>7.2f}"
                f"{np.percentile(hops1, 95):>7.0f}{exact/n:>10.4f}{hitroot/n:>10.4f}{deg:>8.1f}")
            out.append(dict(M=M, leaf_clusters=int(leaf), root_clusters=int(K0), k=kg, mode=mode,
                            hops1=float(np.mean(hops1)), hops3=float(np.mean(hops3)),
                            hop_p95=float(np.percentile(hops1, 95)), exact=exact / n,
                            hit_root=hitroot / n, deg=deg))

    with open(os.path.join(ROOT, "results", "results_vnode_scale.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log("")
    log("  判读：")
    log("   · 同 K0 下 M 增大（副本变多）→ 跳数应基本不变或下降")
    log("   · K0 增大（真类变多）→ 跳数上升；无长链时接近 O(sqrt(M))，加长链后接近 polylog")
    log("   · 把无长链的跳数按 sqrt(M) 外推，就是该结构的规模天花板")


if __name__ == "__main__":
    main()
