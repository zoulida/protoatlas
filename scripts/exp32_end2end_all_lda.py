"""exp32: 端到端总表（判别投影口径）—— 四方对比 + 子原型数扫描，同一次运行

口径问题：exp29（四方）与 exp31（子原型）的路由坐标都取自 16 维主成分，
与正文「坐标维度由类数决定、取满判别方向」不符。本脚本按 LDA d=K-1=9 重做，
并把 A/B/C 与 k=1/2/3/5/8 放在同一次运行里，保证表 5-1 与表 5-2 同源。
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


def train_linear(X, y, n_cls, epochs=300, lr=0.05, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
    return m


def predict(m, X, bs=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.sum():
                C[j] = X[m].mean(0)
    return C


def main():
    from exp12_proto_route import build_split

    torch.manual_seed(0)
    np.random.seed(0)
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"数据：训练 {H.shape} 测试 {T.shape} 节点 {Nn}")

    # ---------------- A 集中式 / B 纯本地 / C FedAvg（384 维输入）
    mA = train_linear(H, y_tr, 10)
    accA = float(np.mean(predict(mA, T).argmax(1) == y_te))
    log(f"A 集中式                                  {accA:.4f}")

    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]), seed=100 + n)
    pred_all = {n: np.array(cls_of_node[n])[predict(heads[n], T).argmax(1)] for n in range(Nn)}
    accB = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))
    log(f"B 纯本地                                  {accB:.4f}")

    torch.manual_seed(200)
    glob = nn.Linear(384, 10)
    gsd = glob.state_dict()
    for r in range(20):
        states = []
        for n in range(Nn):
            m = tr_node == n
            Xn, yn = torch.from_numpy(H[m]), torch.from_numpy(y_tr[m])
            loc = nn.Linear(384, 10)
            loc.load_state_dict(gsd)
            opt = torch.optim.SGD(loc.parameters(), lr=0.05, momentum=0.9)
            lossf = nn.CrossEntropyLoss()
            for _ in range(3):
                opt.zero_grad()
                lossf(loc(Xn), yn).backward()
                opt.step()
            states.append({k: v.clone() for k, v in loc.state_dict().items()})
        gsd = {k: torch.stack([s[k] for s in states]).mean(0) for k in gsd}
    glob.load_state_dict(gsd)
    accC = float(np.mean(predict(glob, T).argmax(1) == y_te))
    log(f"C FedAvg（20 轮 × 3 epoch）                {accC:.4f}")

    # ---------------- D：判别投影 + 子原型数扫描
    mu, W, d_eff = fit_lda(H, y_tr, 9)
    Zq = (T - mu) @ W
    log(f"判别投影 d={d_eff}")
    log(f"  {'每类子原型':>10}{'vnode':>8}{'候选召回@3':>12}{'命中@1':>9}{'端到端':>9}")

    res = {"A_central": accA, "B_local": accB, "C_fedavg": accC,
           "proj": "lda", "d": int(d_eff), "sub": {}}

    for k in (1, 2, 3, 5, 8):
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in nc[n]:
                rows = (tr_node == n) & (y_tr == c)
                X = H[rows]
                if len(X) < k:
                    continue
                Cs = X.mean(0, keepdims=True) if k == 1 else kmeans(X, k, seed=c)
                for v in Cs:
                    protos.append(v)
                    pcls.append(c)
                    pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zp = (P - mu) @ W
        D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                               - 2 * Zq @ Zp.T, 0))
        o = np.argsort(D, 1)

        recall_hit, hit1 = 0, 0
        pred = np.zeros(len(T), np.int32)
        for i in range(len(T)):
            seen, pick = set(), []
            for j in o[i]:
                c = int(pcls[j])
                if c not in seen:
                    seen.add(c)
                    pick.append(j)
                if len(pick) == 3:
                    break
            recall_hit += int(y_te[i] in [int(pcls[j]) for j in pick])
            top = int(pick[0])
            hit1 += int(pcls[top] == y_te[i])
            pred[i] = pred_all[int(pnode[top])][i]
        recall3 = recall_hit / len(T)
        hit1 = hit1 / len(T)
        acc = float(np.mean(pred == y_te))

        log(f"  {k:>10}{len(P):>8}{recall3:>12.4f}{hit1:>9.4f}{acc:>9.4f}")
        res["sub"][str(k)] = dict(n_vnode=len(P), recall3=recall3, hit1=hit1, end2end=acc)

    with open(os.path.join(ROOT, "results", "results_end2end_all_lda.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_end2end_all_lda.json")


if __name__ == "__main__":
    main()
