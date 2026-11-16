"""exp83: 每节点 20 类、每类 300 张（每节点 6000 张）下 body 的作用，noW 口径

对上 exp82（千类、每节点 20 类、每类 25 张）。两档的每节点类数相同，
只差本地样本量，用来回答「样本一多 body 是否有空间」。

设定：CIFAR-100，100 类，5 节点 × 20 类，每类 1 个副本、300 张训练，
测试 3000 张，坐标取 φ(x) 384 维原样。

  E0  body 取恒等
  E1  body = 384→256→ReLU，本地联合训练

除端到端精度外，另报判别侧的分解：
  S_exp  路由送对时本地判对的概率（这才是 body 该改善的量）
  eps    路由送错时蒙对的概率
二者与路由精度 A 的关系为 端到端 = A·S_exp + (1-A)·eps。
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp81_body_np import (init_params, train, predict,  # noqa: E402
                           HID, SEED, EP_JOINT)

NN = 5
PER_NODE = 20
PER_CLASS_TRAIN = 300
N_TEST = 3000


def log(*a):
    print(*a, flush=True)


def main():
    import pandas as pd
    from exp1_route import make_shard2

    t0 = time.time()
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float64)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float64)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr = d["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte = te["fine_label"].to_numpy().astype(np.int64)
    n = min(len(Htr), len(ytr))
    Htr, ytr = Htr[:n], ytr[:n]
    n = min(len(Hte), len(yte))
    Hte, yte = Hte[:n], yte[:n]

    nc, ni = make_shard2(ytr, NN, 100, PER_NODE, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for i, idx in enumerate(ni):
        for c in nc[i]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(PER_CLASS_TRAIN, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), i))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    H, y = Htr[tr_sel], ytr[tr_sel]
    T, yT = Hte[:N_TEST], yte[:N_TEST]
    nc = [sorted(x) for x in nc]
    log(f"训练 {H.shape} 测试 {T.shape} 节点 {NN} 每节点 {len(nc[0])} 类，每节点样本 {len(H)//NN}")

    protos, pcls, pnode, node_z = [], [], [], {}
    for i in range(NN):
        zs = []
        for c in nc[i]:
            p = H[y == c].mean(0)
            protos.append(p)
            pcls.append(c)
            pnode.append(i)
            zs.append(p)
        node_z[i] = np.mean(zs, 0)
    P = np.stack(protos)
    pcls, pnode = np.array(pcls), np.array(pnode)

    Dm = ((T ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2 * T @ P.T)
    np.maximum(Dm, 0, out=Dm)
    top1 = np.argmin(Dm, 1)
    top_node, top_cls = pnode[top1], pcls[top1]
    A = float(np.mean(top_cls == yT))
    log(f"vnode {len(P)}  路由 top-1 {A:.4f}")
    del Dm

    Hn, yn = {}, {}
    for i in range(NN):
        m = tr_node == i
        Hn[i] = H[m]
        yn[i] = np.array([nc[i].index(c) for c in y[m]], dtype=np.int64)

    res = {"n_nodes": NN, "per_node_classes": PER_NODE, "per_class_train": PER_CLASS_TRAIN,
           "route_top1": A}

    def evaluate(tag, ps):
        hit = np.zeros(len(T), bool)
        for i in range(NN):
            m = top_node == i
            if m.sum() == 0:
                continue
            pred = np.array(nc[i])[predict(ps[i], T[m])]
            hit[m] = pred == yT[m]
        right = top_cls == yT
        s_exp = float(hit[right].mean())
        eps = float(hit[~right].mean()) if (~right).sum() else 0.0
        e2e = float(hit.mean())
        res[tag] = {"D1": e2e, "S_exp": s_exp, "eps": eps,
                    "A": A, "check": float(A * s_exp + (1 - A) * eps)}
        log(f"{tag:14s} D1 {e2e:.4f}  送对时判对 {s_exp:.4f}  送错时蒙对 {eps:.4f}  "
            f"分解校验 {res[tag]['check']:.4f}   用时 {time.time()-t0:.0f}s")

    rng = np.random.RandomState(SEED)
    ps = {}
    for i in range(NN):
        ps[i] = init_params(384, None, len(nc[i]), rng)
        train(ps[i], Hn[i], yn[i], len(nc[i]), EP_JOINT)
    evaluate("E0 无 body", ps)

    rng = np.random.RandomState(SEED)
    ps = {}
    for i in range(NN):
        ps[i] = init_params(384, HID, len(nc[i]), rng)
        train(ps[i], Hn[i], yn[i], len(nc[i]), EP_JOINT)
    evaluate("E1 body 本地", ps)

    out = os.path.join(ROOT, "results", "results_body_noW_bigsample.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(out)}")


if __name__ == "__main__":
    main()
