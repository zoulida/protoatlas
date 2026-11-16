# -*- coding: utf-8 -*-
"""类数扫描：K=20/50 的端到端对比（CIFAR-100 前 K 类，384 维坐标，无判别投影）

配方与 exp53 一致，只改类数：节点 2K 个，每节点 2 类，每类 4 副本，每类至多 300 张训练。
K=10 与 K=100 两档已有结果，直接用表 5-1 的读数，本脚本只补中间两档。

用法：KSCAN=20 python exp85_end2end_Kscan.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402

from exp32_end2end_all_lda import fit_lda, train_linear, predict   # noqa: E402
from exp1_route import make_shard2                                # noqa: E402
from exp53_c100_end2end_lda import load_c100, route, ridge_probe   # noqa: E402

K = int(os.environ.get("KSCAN", "20"))
N_NODES = 2 * K
PER_NODE = 2
PER_CLASS_TRAIN = 300
N_TEST = 3000
DIM = 384
SEEDS = (1, 2, 3) if os.environ.get("FAST") == "1" else (1, 2, 3, 4, 5)
ROUNDS = 20
LOCAL_EPOCHS = 3

OUT = os.path.join(ROOT, "results", f"results_Kscan{K}_noW.json")


def log(*a):
    print(*a, flush=True)


def build_split(ytr, k):
    n_nodes = 2 * k
    nc, ni = make_shard2(ytr, n_nodes, k, PER_NODE, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni):
        for c in nc[n]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(PER_CLASS_TRAIN, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    return nc, np.concatenate(tr_sel), np.concatenate(tr_node)


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    H_all, y_all, _, _ = load_c100()

    # 只取前 K 类，标签不变（前 K 类即 0..K-1）
    m_tr = y_all < K
    H_all = H_all[m_tr]
    y_all = y_all[m_tr]

    # 测试集取全量 10 000 张特征，按类均衡抽到约 3 000 张
    T_all = np.load(os.path.join(CACHE, "feat_c100_test10k.npy")).astype(np.float32)
    y_te_all = np.load(os.path.join(CACHE, "c100_test10k.npz"))["y"].astype(np.int64)
    per = max(1, N_TEST // K)
    rng_te = np.random.RandomState(0)
    idx_te = []
    for c in range(K):
        ci = np.where(y_te_all == c)[0]
        idx_te.append(rng_te.choice(ci, size=min(per, len(ci)), replace=False))
    idx_te = np.concatenate(idx_te)
    T = T_all[idx_te]
    y_te = y_te_all[idx_te]

    nc, tr_sel, tr_node = build_split(y_all, K)
    H = H_all[tr_sel]
    y_tr = y_all[tr_sel]
    sizes = np.bincount(tr_node, minlength=N_NODES)
    log(f"K={K} 训练 {H.shape} 测试 {T.shape}")
    log(f"节点 {N_NODES} 每节点 {PER_NODE} 类 每节点样本 {sizes.min()}–{sizes.max()}")

    Wr = ridge_probe(H, y_tr, K)
    Tr = np.hstack([T.astype(np.float64), np.ones((len(T), 1))])
    accR = float(np.mean((Tr @ Wr).argmax(1) == y_te))
    log(f"集中式 ridge（闭式）  {accR:.4f}")

    rec = {"A": [], "B": [], "C": [], "sys": []}
    aux = {"hit1": [], "recall3": [], "sexp": [], "eps": [], "n_vnode": []}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        mA = train_linear(H, y_tr, K, seed=seed)
        accA = float(np.mean(predict(mA, T).argmax(1) == y_te))

        heads = {}
        for n in range(N_NODES):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
        pred_all = {n: np.array(nc[n])[predict(heads[n], T).argmax(1)]
                    for n in range(N_NODES)}
        accB = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(N_NODES)]))

        torch.manual_seed(seed * 1000 + 7)
        glob = nn.Linear(DIM, K)
        gsd = glob.state_dict()
        for _ in range(ROUNDS):
            states = []
            for n in range(N_NODES):
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
        for n in range(N_NODES):
            for c in nc[n]:
                rows = (tr_node == n) & (y_tr == c)
                if rows.sum() == 0:
                    continue
                protos.append(H[rows].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
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
            f"召回@3={r3:.4f} 命中@1={h1:.4f} 端到端={e2e:.4f}")

    out = {"dataset": "cifar100", "K": K, "n_nodes": N_NODES,
           "per_node_classes": PER_NODE, "replicas": N_NODES * PER_NODE // K,
           "d": DIM, "proj": "none", "n_test": len(y_te), "seeds": list(SEEDS),
           "ridge": accR, "acc": {}, "route": {}}

    for key, name in (("A", "同配方集中式"), ("B", "纯本地"), ("C", "FedAvg"),
                      ("sys", "本文")):
        v = np.array(rec[key])
        out["acc"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                               lo=float(v.min()), hi=float(v.max()))
        log(f"{name:<10} {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    for key in aux:
        v = np.array(aux[key], dtype=float)
        out["route"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
    out["acc"]["ridge_closed"] = accR

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("写入", OUT)


if __name__ == "__main__":
    main()
