"""exp54: CIFAR-10 原始像素上的分解（判别投影口径，20 节点 × 2 类）

用途：附录 C 表 C-2 的三行原先取自 exp37/exp38（每类一原型、单点最近原型
判别的口径），其中 CIFAR-10 与 CIFAR-100 两行已按第 5.2 节的口径替换为
exp35 与 exp53 的读数。本脚本把第三行（原始像素，弱特征组）也换成同一口径，
使表内三行同源、可比。

设定与 exp34（表 5-1 的 10 类档）逐处对齐：
  划分、训练子集、测试子集全部取自 exp12_proto_route.build_split()
  20 节点 × 2 类，每类 4 副本，d = 9
  唯一改动是特征：原始像素（逐像素标准化后 PCA-512，与 exp38 的组 2 同）
  而非 DINOv2 的 384 维

输出四个分解量（A、S_exp、eps、E2E）与两条集中式参照（闭式 ridge、梯度训练），
供表 C-2 的第三行使用。
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

from exp32_end2end_all_lda import fit_lda, train_linear, predict  # noqa: E402

K = 10
DIM = 512           # PCA-512 底，3072 维直接算 LDA 太重
SEEDS = (1,) if os.environ.get("ONE_SEED") == "1" else (1, 2, 3, 4, 5)
ROUNDS = 20
LOCAL_EPOCHS = 3

OUT = os.path.join(ROOT, "results", "results_c10_raw_end2end_lda.json")


def log(*a):
    print(*a, flush=True)


def fit_pca(X, d):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = Xc.T @ Xc / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    W = V[:, np.argsort(w)[::-1][:d]]
    return mu, W.astype(np.float32)


def load_raw(tr_sel, te_sel):
    """原始像素 → 标准化 → PCA-512；返回 (训练, 测试) 两组 512 维特征"""
    d10 = np.load(os.path.join(CACHE, "cifar10.npz"))
    Xr = d10["xtr"][tr_sel].reshape(len(tr_sel), -1).astype(np.float32) / 255.0
    Xe = d10["xte"][te_sel].reshape(len(te_sel), -1).astype(np.float32) / 255.0
    m_, s_ = Xr.mean(0, keepdims=True), Xr.std(0, keepdims=True) + 1e-6
    Xr, Xe = (Xr - m_) / s_, (Xe - m_) / s_
    mu_p, W_p = fit_pca(Xr, DIM)
    return (Xr - mu_p) @ W_p, (Xe - mu_p) @ W_p


def route(Zq, Zp, pcls, pnode, pred_all, y_te):
    """全局排序 + 按类去重取最近一类，由宿主节点作答；同时按分解式计数"""
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2.0 * Zq @ Zp.T, 0))
    o = np.argsort(D, 1)
    r3 = h1 = 0
    right_n = right_ok = wrong_n = wrong_ok = 0
    pred = np.zeros(len(Zq), np.int32)
    for i in range(len(Zq)):
        seen, pick = set(), []
        for j in o[i]:
            c = int(pcls[j])
            if c not in seen:
                seen.add(c)
                pick.append(j)
            if len(pick) == 3:
                break
        r3 += int(y_te[i] in [int(pcls[j]) for j in pick])
        top = int(pick[0])
        hit = int(pcls[top] == y_te[i])
        h1 += hit
        p = int(pred_all[int(pnode[top])][i])
        pred[i] = p
        ok = int(p == y_te[i])
        if hit:
            right_n += 1
            right_ok += ok
        else:
            wrong_n += 1
            wrong_ok += ok
    n = len(Zq)
    sexp = right_ok / max(1, right_n)
    eps = wrong_ok / max(1, wrong_n)
    return r3 / n, h1 / n, float(np.mean(pred == y_te)), sexp, eps


def ridge_probe(H, y, n_cls, lam=1.0):
    X = H.astype(np.float64)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.zeros((len(X), n_cls), np.float64)
    Y[np.arange(len(X)), y] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def main():
    from exp12_proto_route import build_split

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H, T = load_raw(tr_sel, te_sel)
    y_tr = ytr[tr_sel].astype(np.int64)
    y_te = yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"训练 {H.shape} 测试 {T.shape} 类数 {len(np.unique(y_tr))} 节点 {Nn}")

    Wr = ridge_probe(H, y_tr, K)
    Tr = np.hstack([T.astype(np.float64), np.ones((len(T), 1))])
    accR = float(np.mean((Tr @ Wr).argmax(1) == y_te))
    log(f"集中式 ridge（闭式，PCA-512 输入）           {accR:.4f}")

    rec = {"A": [], "B": [], "C": [], "sys": []}
    aux = {"hit1": [], "recall3": [], "sexp": [], "eps": [], "n_vnode": []}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        mA = train_linear(H, y_tr, K, seed=seed)
        accA = float(np.mean(predict(mA, T).argmax(1) == y_te))

        heads = {}
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([cls_of_node[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(cls_of_node[n]), seed=seed * 100 + n)
        pred_all = {n: np.array(cls_of_node[n])[predict(heads[n], T).argmax(1)]
                    for n in range(Nn)}
        accB = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))

        torch.manual_seed(seed * 1000 + 7)
        glob = nn.Linear(DIM, K)
        gsd = glob.state_dict()
        for _ in range(ROUNDS):
            states = []
            for n in range(Nn):
                m = tr_node == n
                Xn, yn = torch.from_numpy(H[m]), torch.from_numpy(y_tr[m])
                loc = nn.Linear(DIM, K)
                loc.load_state_dict(gsd)
                opt = torch.optim.SGD(loc.parameters(), lr=0.05, momentum=0.9)
                lossf = nn.CrossEntropyLoss()
                for _ in range(LOCAL_EPOCHS):
                    opt.zero_grad()
                    lossf(loc(Xn), yn).backward()
                    opt.step()
                states.append({k: v.clone() for k, v in loc.state_dict().items()})
            gsd = {k: torch.stack([s[k] for s in states]).mean(0) for k in gsd}
        glob.load_state_dict(gsd)
        accC = float(np.mean(predict(glob, T).argmax(1) == y_te))

        mu, W, d_eff = fit_lda(H, y_tr, K - 1)
        Zq = (T - mu) @ W
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in cls_of_node[n]:
                rows = (tr_node == n) & (y_tr == c)
                if rows.sum() == 0:
                    continue
                protos.append(H[rows].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls = np.array(pcls)
        pnode = np.array(pnode)
        Zp = (P - mu) @ W
        r3, h1, e2e, sexp, eps = route(Zq, Zp, pcls, pnode, pred_all, y_te)

        rec["A"].append(accA)
        rec["B"].append(accB)
        rec["C"].append(accC)
        rec["sys"].append(e2e)
        aux["hit1"].append(h1)
        aux["recall3"].append(r3)
        aux["sexp"].append(sexp)
        aux["eps"].append(eps)
        aux["n_vnode"].append(len(P))
        log(f"种子 {seed}: A={accA:.4f} B={accB:.4f} C={accC:.4f} | "
            f"召回@3={r3:.4f} 命中@1={h1:.4f} 端到端={e2e:.4f} | "
            f"S_exp={sexp:.4f} eps={eps:.4f} d_eff={d_eff}")

    out = {"dataset": "cifar10_raw_pca512", "K": K, "n_nodes": Nn,
           "per_node_classes": C["shard_per_node"], "d": K - 1, "proj": "lda",
           "n_test": int(len(y_te)), "seeds": list(SEEDS),
           "ridge": accR, "acc": {}, "route": {}}

    log("")
    for key, name in (("A", "同配方集中式（梯度）"), ("B", "纯本地"),
                      ("C", "FedAvg"), ("sys", "本文（k=1）")):
        v = np.array(rec[key])
        out["acc"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                               lo=float(v.min()), hi=float(v.max()))
        log(f"{name}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    log(f"ridge（闭式）: {accR:.4f}")
    for key in ("recall3", "hit1", "sexp", "eps", "n_vnode"):
        v = np.array(aux[key], dtype=float)
        out["route"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
        log(f"{key}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    log(f"核对 A·S_exp + (1-A)·eps = "
        f"{out['route']['hit1']['mean'] * out['route']['sexp']['mean'] + (1 - out['route']['hit1']['mean']) * out['route']['eps']['mean']:.4f}"
        f"  实测 E2E {out['acc']['sys']['mean']:.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
