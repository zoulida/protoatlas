"""exp7: IID(shard-2) 下 vnode 重合度诊断
回答：同类各节点的原型挤在一起会怎样？用数字说话。

指标：
  intra/inter            同类异节点距离 / 异类距离（重合的直接度量）
  overlap_rate           同类 vnode 对的间距 > 最近异类间距 的比例（真重叠）
  snr                    原型偏移 / 采样噪声（≈1 说明分散全是噪声）
  conn / diam            KNN 图连通分量、直径
  load_top3              最热 3 个 vnode 吃掉的查询比例
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DIM = 16
K = 8


def log(*a):
    print(*a, flush=True)


def load_aligned():
    sys.path.insert(0, ROOT)
    import exp1_route as E
    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    nc, ni = E.make_shard2(ytr, C["n_nodes"], C["n_classes"],
                           C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    sel, nod = [], []
    for n, idx in enumerate(ni):
        for k in nc[n]:
            sub = idx[ytr[idx] == k]
            take = rng.choice(sub, size=min(C["per_class_train"], len(sub)), replace=False)
            sel.append(take)
            nod.append(np.full(len(take), n))
    tr_sel = np.concatenate(sel)
    tr_node = np.concatenate(nod)
    te_sel = rng.choice(len(xte), size=min(C["n_test"], len(xte)), replace=False)
    return dict(y=ytr[tr_sel], tr_node=tr_node, te_sel=te_sel, yte_al=yte[te_sel])


def bfs_diam(nb, V):
    import collections
    best = 0
    for s in range(V):
        dist = {s: 0}
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if len(dist) == V:
            best = max(best, max(dist.values()))
        else:
            return -1
    return best


def main():
    t0 = time.time()
    P = load_aligned()
    Htr = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    Hte = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    assert len(Htr) == len(P["y"]) == len(P["tr_node"])

    mu = Htr.mean(0, keepdims=True)
    Hc = Htr - mu
    cov = (Hc.T @ Hc) / (len(Hc) - 1)
    w, V = np.linalg.eigh(cov.astype(np.float64))
    W = V[:, ::-1][:, :DIM].astype(np.float32)
    Z = Hc @ W
    Zte = (Hte - mu) @ W
    y, nod = P["y"], P["tr_node"]

    # ---- 每 (节点, 类) 原型
    keys = sorted(set(zip(nod.tolist(), y.tolist())))
    Pv = np.stack([Z[(nod == n) & (y == k)].mean(0) for n, k in keys])
    cnt = np.array([int(((nod == n) & (y == k)).sum()) for n, k in keys])
    Vn = len(keys)
    log(f"vnode 数 {Vn}，每节点类数 {dict((n, sum(1 for a,b in keys if a==n)) for n in range(10))}")
    log(f"每 (节点,类) 样本数 {cnt.min()}~{cnt.max()}")

    # ---- 类内/类间距离
    D = np.linalg.norm(Pv[:, None, :] - Pv[None, :, :], axis=-1)
    same = np.array([keys[a][1] == keys[b][1] for a in range(Vn) for b in range(Vn)]).reshape(Vn, Vn)
    iu = np.triu_indices(Vn, 1)
    intra = D[iu][same[iu]]
    inter = D[iu][~same[iu]]
    # 真重叠：某个同类 vnode 对的距离，比"该对中任一方最近的异类 vnode"还远
    np.fill_diagonal(D, np.inf)
    cls_arr = np.array([k for _, k in keys])
    ov = 0
    for a in range(Vn):
        for b in range(Vn):
            if b <= a or keys[a][1] != keys[b][1]:
                continue
            near_other = min(D[a][cls_arr != keys[a][1]].min(),
                             D[b][cls_arr != keys[a][1]].min())
            if D[a][b] > near_other:
                ov += 1
    n_same_pairs = len(intra)

    # ---- snr：原型偏移 vs 采样噪声
    snrs = []
    cls_mu = {k: Z[y == k].mean(0) for k in range(10)}
    for i, (n, k) in enumerate(keys):
        pts = Z[(nod == n) & (y == k)]
        r_k = float(np.sqrt(((pts - cls_mu[k]) ** 2).sum(1).mean()))
        bias = float(np.linalg.norm(Pv[i] - cls_mu[k]))
        snrs.append(bias / (r_k / np.sqrt(len(pts))))

    # ---- KNN 图
    nb = [list(np.argsort(D[a])[:K]) for a in range(Vn)]
    adj = [set() for _ in range(Vn)]
    for a in range(Vn):
        for b in nb[a]:
            adj[a].add(b); adj[b].add(a)
    seen = np.zeros(Vn, bool); comps = []
    for s in range(Vn):
        if seen[s]:
            continue
        stack, c = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop(); c += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v)
        comps.append(c)
    comps.sort(reverse=True)
    diam = bfs_diam(adj, Vn)
    cross = sum(1 for a in range(Vn) for b in adj[a] if keys[a][1] != keys[b][1])
    intra_edges = sum(len(adj[a]) for a in range(Vn)) - cross

    # ---- 检索命中集中度
    Dq = ((Zte[:, None, :] - Pv[None, :, :]) ** 2).sum(-1)
    order = np.argsort(Dq, axis=1)
    vclass = np.array([keys[i][1] for i in range(Vn)])
    hit1 = float(np.mean(vclass[order[:, 0]] == P["yte_al"]))
    hit3 = float(np.mean([P["yte_al"][i] in vclass[order[i, :3]] for i in range(len(Zte))]))
    cnt_hit = np.bincount(order[:, 0], minlength=Vn) / len(Zte)
    top3_load = float(np.sort(cnt_hit)[::-1][:3].sum())
    gini = 1 - 2 * np.trapezoid(np.cumsum(np.sort(cnt_hit)) / np.sum(cnt_hit),
                                dx=1 / Vn)

    top_idx = np.argsort(cnt_hit)[::-1][:3]
    res = dict(
        n_vnode=Vn,
        intra_mean=float(intra.mean()), intra_min=float(intra.min()), intra_max=float(intra.max()),
        inter_mean=float(inter.mean()), inter_min=float(inter.min()),
        ratio=float(intra.mean() / inter.mean()),
        overlap_pairs=int(ov), overlap_rate=float(ov / n_same_pairs),
        snr_med=float(np.median(snrs)), snr_p90=float(np.percentile(snrs, 90)),
        n_comp=len(comps), largest_frac=comps[0] / Vn, diam=int(diam),
        avg_deg=float(np.mean([len(a) for a in adj])),
        cross_edge_frac=float(cross / max(cross + intra_edges, 1)),
        hit1=hit1, hit3=hit3, load_top3=top3_load, load_gini=float(gini),
        top3_vnodes=[f"{keys[int(i)]}:{cnt_hit[i]*100:.1f}%" for i in top_idx],
    )

    log("=" * 60)
    log(f"同类异节点距离  mean {res['intra_mean']:.3f}  min {res['intra_min']:.3f}  max {res['intra_max']:.3f}")
    log(f"异类距离        mean {res['inter_mean']:.3f}  min {res['inter_min']:.3f}")
    log(f"比值 intra/inter = {res['ratio']:.3f}")
    log(f"真重叠对数      {res['overlap_pairs']}/{n_same_pairs} = {res['overlap_rate']*100:.1f}%")
    log(f"偏置/噪声 snr   中位 {res['snr_med']:.2f}  p90 {res['snr_p90']:.2f}")
    log(f"KNN 图          分量 {res['n_comp']}（最大 {res['largest_frac']*100:.0f}%）  直径 {res['diam']}  平均度 {res['avg_deg']:.1f}  跨类边 {res['cross_edge_frac']*100:.0f}%")
    log(f"检索            hit@1 {hit1:.3f}  hit@3 {hit3:.3f}")
    log(f"负载            最热 3 个 vnode 占 {top3_load*100:.1f}%   Gini {gini:.3f}")
    log(f"                最热 {res['top3_vnodes']}")
    with open(os.path.join(ROOT, "results", "results_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"完成 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
