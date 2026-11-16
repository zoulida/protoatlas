"""exp29b: 端到端四方对比 —— 按判别投影（LDA, d=K-1）重做路由一侧

exp29 的路由坐标取自 16 维主成分投影，与正文第 4.2.3 节「坐标维度由类数决定、
取满判别方向」不一致，端到端的本文读数因此偏低。本脚本保持 A/B/C 三个分类器
不变（均为 384 维输入，与路由投影无关），只把 D 档的候选生成改到判别投影上，
并同时给出单原型与 8 子原型两种配置。
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

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    if os.environ.get("NORM"):
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"数据：训练 {H.shape} 测试 {T.shape} 节点 {Nn}")

    # ---------------- A/B/C：384 维输入，与路由投影无关
    mA = train_linear(H, y_tr, 10)
    accA = float(np.mean(predict(mA, T).argmax(1) == y_te))
    log(f"A 集中式                                  {accA:.4f}")

    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]))

    def local_pred(n, X):
        lg = predict(heads[n], X)
        return np.array(cls_of_node[n])[lg.argmax(1)], lg.max(1)

    accB = float(np.mean([np.mean(local_pred(n, T)[0] == y_te) for n in range(Nn)]))
    log(f"B 纯本地                                  {accB:.4f}")

    torch.manual_seed(0)
    glob = nn.Linear(384, 10)
    gsd = glob.state_dict()
    for r in range(20):
        states = []
        for n in range(Nn):
            m = tr_node == n
            Xn = torch.from_numpy(H[m])
            yn = torch.from_numpy(y_tr[m])
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

    mu, W, d_eff = fit_lda(H, y_tr, 9)
    log(f"判别投影 d={d_eff}")
    Zq = (T - mu) @ W

    pred_all = {n: local_pred(n, T)[0] for n in range(Nn)}
    res = {"A_central": accA, "B_local": accB, "C_fedavg": accC, "d": int(d_eff)}

    for k_sub in (1, 8):
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            m = tr_node == n
            for c in nc[n]:
                rows = m & (y_tr == c)
                X = H[rows]
                if len(X) == 0:
                    continue
                if k_sub == 1:
                    Cs = X.mean(0, keepdims=True)
                else:
                    Cs = kmeans(X, min(k_sub, len(X)))
                for v in Cs:
                    protos.append(v)
                    pcls.append(c)
                    pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zp = (P - mu) @ W
        D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                               - 2 * Zq @ Zp.T, 0))
        order = np.argsort(D, 1)

        def cands(i, k=3):
            seen, out = set(), []
            for j in order[i]:
                c = int(pcls[j])
                if c not in seen:
                    seen.add(c)
                    out.append((c, int(pnode[j]), float(D[i, j])))
                if len(out) == k:
                    break
            return out

        cl = [cands(i) for i in range(len(T))]
        in_cand = np.array([y_te[i] in [c for c, _, _ in cl[i]] for i in range(len(T))])
        d1 = np.array([pred_all[cl[i][0][1]][i] for i in range(len(T))])
        accD1 = float(np.mean(d1 == y_te))

        d2 = []
        for i in range(len(T)):
            cs = cl[i]
            ds = np.array([c[2] for c in cs])
            wts = np.exp(-ds / (ds.mean() + 1e-8))
            sc = np.zeros(10)
            for (c, n, dd), wt in zip(cs, wts):
                sc[pred_all[n][i]] += wt
            d2.append(int(np.argmax(sc)))
        accD2 = float(np.mean(np.array(d2) == y_te))
        accD3 = float(np.mean([any(pred_all[n][i] == y_te[i] for c, n, _ in cl[i])
                               for i in range(len(T))]))

        tag = "single" if k_sub == 1 else f"sub{k_sub}"
        res[tag] = dict(n_vnode=len(P), D1=accD1, D2=accD2, D3_oracle=accD3,
                        cand_recall3=float(in_cand.mean()))
        log(f"  [{tag}] vnode {len(P):4d}  候选召回@3 {in_cand.mean():.4f}  "
            f"D1 {accD1:.4f}  D2 {accD2:.4f}  D3 {accD3:.4f}")

    with open(os.path.join(ROOT, "results", "results_end2end_lda.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_end2end_lda.json")


if __name__ == "__main__":
    main()
