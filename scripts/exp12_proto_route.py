"""exp12: 原型级路由准确率（用从未参与训练的测试集做查询）

前的 0.947 是"单样本 1-NN"，不是真实路由：真实路由的对象是 vnode 原型
（每节点每类的均值），且查询应来自干净的测试集。本脚本给出论文该报的数字。

另外测一个 IID 下的陷阱：top-3 会被"同类的重复原型"占满吗？
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


def d2mat(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def build_split():
    import exp1_route as E

    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    nc, ni = E.make_shard2(ytr, C["n_nodes"], C["n_classes"], C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni):
        for k in nc[n]:
            sel = idx[ytr[idx] == k]
            take = rng.choice(sel, size=min(C["per_class_train"], len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    te_sel = rng.choice(len(xte), size=min(C["n_test"], len(xte)), replace=False)
    return nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C


def main():
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    assert len(H) == len(tr_sel) and len(T) == len(te_sel)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]

    # 构造 vnode 原型：每 (节点, 类) 一个
    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            if rows.sum() == 0:
                continue
            protos.append(H[rows].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls = np.array(pcls)
    pnode = np.array(pnode)
    log(f"vnode 原型数 {len(P)}（{C['n_nodes']} 节点 × {C['shard_per_node']} 类）")
    log(f"每类覆盖节点数 {np.bincount(pcls, minlength=10).tolist()}")

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)

    res = {}
    for name, dim in [("全维 384", 384), ("PCA-16", 16)]:
        Wk = np.eye(384, dtype=np.float32) if dim == 384 else W
        Zq = (T - mu) @ Wk
        Zp = (P - mu) @ Wk
        D = np.sqrt(d2mat(Zq, Zp))
        order = np.argsort(D, axis=1)

        hit1 = float(np.mean(pcls[order[:, 0]] == y_te))
        # top-k 去重到"不同类"
        def dedup_hit(k):
            hit, ndist = 0, []
            for i in range(len(y_te)):
                seen, picked = set(), []
                for j in order[i]:
                    c = pcls[j]
                    if c not in seen:
                        seen.add(c)
                        picked.append(c)
                    if len(picked) == k:
                        break
                ndist.append(len(seen))
                hit += int(y_te[i] in picked)
            return hit / len(y_te), float(np.mean(ndist))

        h3, n3 = dedup_hit(3)
        h5, n5 = dedup_hit(5)
        # top-3 原始（不去重）的不同类数
        raw3 = float(np.mean([len(set(pcls[order[i, :3]])) for i in range(len(y_te))]))

        res[name] = dict(hit1=hit1, hit3_dedup=h3, hit5_dedup=h5, mean_distinct_in_top3=raw3)
        log(f"\n[{name}]")
        log(f"  命中@1（最近原型的类对不对）      {hit1:.4f}")
        log(f"  top-3 原始候选里平均有几个不同类   {raw3:.2f}")
        log(f"  按类去重取前 3 类，命中            {h3:.4f}（平均取到 {n3:.2f} 个不同类）")
        log(f"  按类去重取前 5 类，命中            {h5:.4f}（平均取到 {n5:.2f} 个不同类）")

    with open(os.path.join(ROOT, "results", "results_proto_route.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
