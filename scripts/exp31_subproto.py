"""exp31: 每类多个子原型，能不能把路由命中提上去？

现状：每类 1 个原型（全部样本的均值）→ hit@1 0.9448，端到端 0.9448。
若类内是多模态的（一类的样本聚成几团），单个均值会落在团与团之间，
查询到它的距离反而变远。

测每类 k = 1 / 2 / 3 / 5 / 8 个子中心（k-means）：
  ① hit@1（路由到正确的类）
  ② 端到端（路由 + 节点 head）
  ③ 每节点广播的原型数（通信成本）
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        D = ((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T)
        a = D.argmin(1)
        for j in range(k):
            m = a == j
            if m.sum() > 0:
                C[j] = X[m].mean(0)
    return C


def train_linear(X, y, n_cls, epochs=300, lr=0.05, wd=1e-4):
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
    return m


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]))

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zq = (T - mu[0]) @ W

    def head_pred(n, idx):
        with torch.no_grad():
            lg = heads[n](torch.from_numpy(T[idx])).numpy()
        return np.array(cls_of_node[n])[lg.argmax(1)]

    log(f"  {'每类子原型数':>12}{'原型总数':>10}{'hit@1':>9}{'端到端':>9}")
    log("  " + "-" * 42)
    res = {}
    for k in [1, 2, 3, 5, 8]:
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in nc[n]:
                m = tr_node == n
                X = H[m & (y_tr == c)]
                if len(X) < k:
                    continue
                if k == 1:
                    Cs = X.mean(0, keepdims=True)
                else:
                    Cs = kmeans(X, k, seed=c)
                for cc in Cs:
                    protos.append(cc)
                    pcls.append(c)
                    pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zp = (P - mu[0]) @ W
        D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                               - 2 * Zq @ Zp.T, 0))
        o = D.argmin(1)
        hit = float(np.mean(pcls[o] == y_te))

        # 端到端：最近的子原型 → 宿主节点 → head
        pred = np.zeros(len(T), np.int32)
        for n in np.unique(pnode[o]):
            idx = np.where(pnode[o] == n)[0]
            pred[idx] = head_pred(n, idx)
        acc = float(np.mean(pred == y_te))
        log(f"  {k:>12}{len(protos):>10}{hit:>9.4f}{acc:>9.4f}")
        res[k] = dict(n_proto=len(protos), hit1=hit, end2end=acc,
                      per_node=len(protos) / Nn)

    log("\n  参考：集中式 0.9630 / FedAvg 0.9672 / 单原型端到端 0.9448")
    with open(os.path.join(ROOT, "results", "results_subproto.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
