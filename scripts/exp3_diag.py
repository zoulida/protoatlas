# -*- coding: utf-8 -*-
"""
诊断：路由失败到底出在"筛候选"还是"排序"？
并比较三种降到 2 维的方式：随机投影 / PCA / LDA

关键新指标：
  recall@cand = 真实类所属的节点，被候选池保留的比例   （筛子好不好）
  hit@1       = 候选池里排序后 top1 命中的比例          （排序好不好）
"""
import os, json, time
import numpy as np
from collections import defaultdict
import exp2_sweep as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = E.log


def project(H_ref, H_q, kind, dim, y_ref=None):
    if kind == "random":
        R = np.random.RandomState(0).randn(dim, H_ref.shape[1]) / np.sqrt(H_ref.shape[1])
        mu = H_ref.mean(0, keepdims=True)
        return (H_ref - mu) @ R.T, (H_q - mu) @ R.T
    if kind == "pca":
        mu = H_ref.mean(0, keepdims=True)
        A = H_ref - mu
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        W = Vt[:dim].T
        a, b = A @ W, (H_q - mu) @ W
        # 逐维标准化到可比的尺度
        sd = a.std(0, keepdims=True) + 1e-9
        return a / sd, b / sd
    if kind == "lda":
        classes = np.unique(y_ref)
        mu_all = H_ref.mean(0)
        Sw = np.zeros((H_ref.shape[1], H_ref.shape[1]))
        means = []
        for c in classes:
            m = y_ref == c
            mc = H_ref[m].mean(0)
            means.append(mc)
            Sw += (H_ref[m] - mc).T @ (H_ref[m] - mc)
        Sw /= len(H_ref)
        means = np.stack(means)
        Sb = np.cov(means.T)
        Sb = Sb + np.eye(Sb.shape[0]) * (1e-6 * np.trace(Sb) / Sb.shape[0])
        # 用 Sw^-1 Sb 的主特征向量（Sw 用正则化伪逆，避免奇异）
        lam = 1e-2 * np.trace(Sw) / Sw.shape[0]
        W = np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), Sb)
        # 取 W 做投影：用 SVD 提主方向
        Uw, sv, Vwt = np.linalg.svd(W, full_matrices=False)
        Wp = Vwt[:dim].T
        a, b = (H_ref - mu_all) @ Wp, (H_q - mu_all) @ Wp
        sd = a.std(0, keepdims=True) + 1e-9
        return a / sd, b / sd
    raise ValueError(kind)


def quantize_to_unit(a, b):
    """用 a 的分位数表把 a、b 映射到 [0,1]"""
    out_a = np.zeros_like(a)
    out_b = np.zeros_like(b)
    for j in range(a.shape[1]):
        qs = np.quantile(a[:, j], np.linspace(0, 1, 1001))
        qg = np.linspace(0, 1, 1001)
        out_a[:, j] = np.interp(a[:, j], qs, qg)
        out_b[:, j] = np.interp(b[:, j], qs, qg)
    return np.clip(out_a, 0, 1), np.clip(out_b, 0, 1)


def evaluate(H_tr, H_te, y_tr, y_te, node_classes, node_pos, pos_map,
             kind, dim, grid, tau, ring=0, topk=3):
    a, b = project(H_tr, H_te, kind, dim, y_tr)
    z_tr, z_te = quantize_to_unit(a, b)
    cells_tr = E.to_cells(z_tr, grid)
    reg = E.build_registry(H_tr, y_tr, cells_tr, node_classes, node_pos, pos_map, tau)

    node_cls_set = [set(cs) for cs in node_classes]
    # 每个类有哪些节点
    cls_nodes = defaultdict(set)
    for n, cs in enumerate(node_classes):
        for k in cs:
            cls_nodes[k].add(n)

    n = len(H_te)
    h1 = h3 = 0
    recall_num = recall_den = 0
    cand_sum = 0
    empty = 0
    for i in range(n):
        z = z_te[i]
        c = (int(min(max(z[0] * grid, 0), grid - 1e-9)),
             int(min(max(z[1] * grid, 0), grid - 1e-9)))
        cand = list(reg.get(c, []))
        if ring > 0 and len(cand) < 3:
            for cc in E.neighbors_ring(c, grid, ring):
                cand.extend(reg.get(cc, []))
        cand_nodes = {e[0] for e in cand}
        cand_sum += len(cand_nodes)
        y = int(y_te[i])
        truth = cls_nodes[y]
        recall_num += len(truth & cand_nodes)
        recall_den += len(truth)
        if not cand_nodes:
            empty += 1
            continue
        best = {}
        for node, k, proto, ncnt in cand:
            d = float(np.dot(H_te[i] - proto, H_te[i] - proto))
            if node not in best or d < best[node]:
                best[node] = d
        order = [nd for nd, _ in sorted(best.items(), key=lambda kv: kv[1])][:topk]
        if y in node_cls_set[order[0]]:
            h1 += 1
        if any(y in node_cls_set[nd] for nd in order):
            h3 += 1

    return dict(kind=kind, dim=dim, grid=grid, ring=ring,
                hit1=h1 / n, hit3=h3 / n,
                recall=recall_num / recall_den,
                cand_mean=cand_sum / n, empty=empty / n,
                cells=len(reg))


def main():
    log("=" * 78)
    log("诊断实验：投影方式 × 维度")
    xtr, ytr, xte, yte = E.load_cifar()
    node_classes, node_idx = E.make_shard2(ytr, 20, 10, 2, 0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(node_idx):
        for k in node_classes[n]:
            sel = idx[ytr[idx] == k]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    te_sel = rng.choice(len(xte), size=5000, replace=False)

    H_tr = np.load(os.path.join(E.CACHE, "feat_train.npy"))
    H_te = np.load(os.path.join(E.CACHE, "feat_test.npy"))
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    node_pos = [tr_sel[tr_node == n] for n in range(20)]
    pos_map = {int(s): i for i, s in enumerate(tr_sel)}

    rows = []
    for kind in ["random", "pca", "lda"]:
        for dim in [2, 8]:
            for grid in [4, 8, 16]:
                r = evaluate(H_tr, H_te, y_tr, y_te, node_classes, node_pos, pos_map,
                             kind, dim, grid, 5, ring=0)
                rows.append(r)
                log(f"  {kind:<7} d={dim} G={grid:<3} "
                    f"筛召回={r['recall']:.3f} 候选={r['cand_mean']:.1f} "
                    f"hit@1={r['hit1']:.3f} hit@3={r['hit3']:.3f} "
                    f"空={r['empty']:.3f} 格={r['cells']}/{grid**dim if dim <= 3 else '?'}")

    # 上界
    fs1, fs3 = E.fullscan(H_tr, H_te, y_tr, y_te, node_classes, node_pos, pos_map)
    log("")
    log(f"全扫描上界 hit@1={fs1:.3f} hit@3={fs3:.3f}")

    with open(os.path.join(ROOT, "results", "results_diag.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, fullscan_hit1=fs1, fullscan_hit3=fs3),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
