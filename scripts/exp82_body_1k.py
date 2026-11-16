"""exp82: 千类档（K=1000、50 节点 × 20 类）下 body 有没有用，noW 口径

与 exp68 同设定，但坐标取 φ(x) 的 384 维原样（不作投影），并按正文的
「一组坐标邻近的节点共同训练」补上簇内同步一档。

数据：ImageNet-1k 验证集特征，每类 50 张，前 25 张训练、后 25 张测试。
每节点持有 20 类、500 张，每类只有 1 个副本，故没有「同类节点」可聚，
簇内同步按节点坐标（该节点 20 个原型的均值）取 8 近邻。

  E0  body 取恒等，head 直接接 384 维
  E1  body = 384→256→ReLU，本地联合训练
  E2  body 同 E1，与坐标最近的 8 个节点同步 3 轮

读两个量：D1（top-1 候选的宿主作答后的端到端精度）与路由 top-1。
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp81_body_np import (init_params, train, train_head_only, predict,  # noqa: E402
                           HID, SEED, EP_JOINT, EP_SYNC, EP_HEAD, ROUNDS)

K = 1000
NN = 50
N_TR = 25


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float64)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    tr_idx, te_idx = [], []
    for c in range(K):
        rows = np.where(yt == c)[0]
        tr_idx.append(rows[:N_TR])
        te_idx.append(rows[N_TR:])
    H, y = Ht[np.concatenate(tr_idx)], yt[np.concatenate(tr_idx)]
    T, yT = Ht[np.concatenate(te_idx)], yt[np.concatenate(te_idx)]
    nc = [list(map(int, a)) for a in np.array_split(np.arange(K), NN)]
    log(f"训练 {H.shape} 测试 {T.shape} 节点 {NN} 每节点 {len(nc[0])} 类")

    # 原型：每 (节点, 类) 一个，共 1000 个；每类仅 1 个副本
    protos, pcls, pnode, node_z = [], [], [], {}
    for n in range(NN):
        zs = []
        for c in nc[n]:
            p = H[y == c].mean(0)
            protos.append(p)
            pcls.append(c)
            pnode.append(n)
            zs.append(p)
        node_z[n] = np.mean(zs, 0)
    P = np.stack(protos)
    pcls, pnode = np.array(pcls), np.array(pnode)

    # 路由：坐标原样，最近原型即 top-1
    Dm = ((T ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2 * T @ P.T)
    np.maximum(Dm, 0, out=Dm)
    top1 = np.argmin(Dm, 1)
    top_node, top_cls = pnode[top1], pcls[top1]
    route = float(np.mean(top_cls == yT))
    log(f"vnode {len(P)}  路由 top-1 {route:.4f}   用时 {time.time()-t0:.0f}s")
    del Dm

    NZ = np.stack([node_z[n] for n in range(NN)])
    dnn = ((NZ[:, None, :] - NZ[None, :, :]) ** 2).sum(-1)
    nbr8 = {n: list(np.argsort(dnn[n])[:8]) for n in range(NN)}

    Hn, yn = {}, {}
    for n in range(NN):
        m = np.isin(y, nc[n])
        Hn[n] = H[m]
        yn[n] = np.array([nc[n].index(c) for c in y[m]], dtype=np.int64)

    res = {"K": K, "n_nodes": NN, "route_top1": route, "vnode": int(len(P))}
    cnt = {n: int((top_node == n).sum()) for n in range(NN)}

    def evaluate(tag, ps):
        right = 0
        for n in range(NN):
            m = top_node == n
            if m.sum() == 0:
                continue
            pred = np.array(nc[n])[predict(ps[n], T[m])]
            right += int((pred == yT[m]).sum())
        res[tag] = right / len(T)
        log(f"{tag:22s} D1 {res[tag]:.4f}   用时 {time.time()-t0:.0f}s")

    rng = np.random.RandomState(SEED)
    ps = {}
    for n in range(NN):
        ps[n] = init_params(384, None, len(nc[n]), rng)
        train(ps[n], Hn[n], yn[n], len(nc[n]), EP_JOINT)
    evaluate("E0 无 body", ps)

    rng = np.random.RandomState(SEED)
    ps = {}
    for n in range(NN):
        ps[n] = init_params(384, HID, len(nc[n]), rng)
        train(ps[n], Hn[n], yn[n], len(nc[n]), EP_JOINT)
    evaluate("E1 body 本地", ps)

    rng = np.random.RandomState(SEED)
    base = init_params(384, HID, 2, rng)
    ps = {}
    for n in range(NN):
        ps[n] = init_params(384, HID, len(nc[n]), np.random.RandomState(SEED))
        ps[n]["W1"], ps[n]["b1"] = base["W1"].copy(), base["b1"].copy()
    for r in range(ROUNDS):
        for n in range(NN):
            train(ps[n], Hn[n], yn[n], len(nc[n]), EP_SYNC)
        W1, b1 = {}, {}
        for n in range(NN):
            W1[n] = np.mean([ps[j]["W1"] for j in nbr8[n]], 0)
            b1[n] = np.mean([ps[j]["b1"] for j in nbr8[n]], 0)
        for n in range(NN):
            ps[n]["W1"], ps[n]["b1"] = W1[n], b1[n]
            k = len(nc[n])
            ps[n]["W2"] = np.random.RandomState(SEED + n).normal(0, np.sqrt(1.0 / HID), (HID, k))
            ps[n]["b2"] = np.zeros(k)
            train_head_only(ps[n], Hn[n], yn[n], k, EP_HEAD)
        log(f"   E2 第 {r+1} 轮完成   用时 {time.time()-t0:.0f}s")
    evaluate("E2 body 近邻同步(8x3)", ps)

    res["cnt_min"] = min(cnt.values())
    res["cnt_max"] = max(cnt.values())
    out = os.path.join(ROOT, "results", "results_body_noW_1k.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(out)}")


if __name__ == "__main__":
    main()
