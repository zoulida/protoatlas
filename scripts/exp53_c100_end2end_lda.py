"""exp53: CIFAR-100 上的端到端四方对比（判别投影口径，200 节点 × 2 类）

背景：表 5-1 的四路对比只在 CIFAR-10 上做过，CIFAR-100 一侧只有路由层读数
（5.2.1 的召回与询问量、表 C-2 的 A 对 G_lin）。第 6 章却写了「在 CIFAR-10
与 CIFAR-100 的设定下，端到端精度与集中式处于同一量级」，缺 C-100 的端到端
读数，本脚本补这一档。

设定对齐：
  K = 100 类，D = 384，d = min(K-1, D) = 99
  200 节点 × 2 类 = 400 槽位 / 100 类 = 每类 4 个副本（与表 5-1 的 C-10 设定
    同为 R=4，故两个数据集上只有类数变，副本数不变）
  每 (节点,类) 取 min(300, 该类可用样本/4) 张；CIFAR-100 每类 500 张训练，
    故每节点实际 2×125=250 张（C-10 为 2×300=600 张，这一差别由数据量决定，
    不可消除，报告时须注明）
  测试 5 000 张，与表 5-1 同

口径：与 exp34 完全一致，坐标走判别投影、vnode 每 (节点,类) 一个、全局排序后
按类去重取最近的一类、由该类所在的宿主节点用本地 head 作答。四条读数：
  A 同配方集中式（线性头，全部数据）
  B 纯本地（每节点一个 head，只在本地 2 类上训练）
  C FedAvg（服务器 + 20 轮 × 3 epoch 本地 SGD 后参数平均）
  ridge 闭式解（同族最强，无随机性）
  本文 k=1
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp32_end2end_all_lda import fit_lda, train_linear, predict  # noqa: E402
from exp1_route import make_shard2  # noqa: E402

K = 100
N_NODES = 200
PER_NODE_CLASSES = 2
PER_CLASS_TRAIN = 300
N_TEST = 3000  # 测试特征仅缓存了 3000 张（5.2.1 同此），与 C-10 的 5000 张不同
DIM = 384
SEEDS = (1,) if os.environ.get("ONE_SEED") == "1" else (1, 2, 3, 4, 5)
ROUNDS = 20
LOCAL_EPOCHS = 3

OUT = os.path.join(ROOT, "results", "results_c100_end2end_lda.json")


def log(*a):
    print(*a, flush=True)


def load_c100():
    import pandas as pd
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr = d["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte = te["fine_label"].to_numpy().astype(np.int64)
    n_te = min(len(Hte), len(yte))
    Hte, yte = Hte[:n_te], yte[:n_te]
    n_tr = min(len(Htr), len(ytr))
    return Htr[:n_tr], ytr[:n_tr], Hte, yte


def build_split(ytr):
    """200 节点 × 2 类，每类 4 副本；返回 (每节点类表, 选中行, 行所属节点)"""
    nc, ni = make_shard2(ytr, N_NODES, K, PER_NODE_CLASSES, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni):
        for c in nc[n]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(PER_CLASS_TRAIN, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    return [sorted(x) for x in nc], tr_sel, tr_node


def route(Zq, Zp, pcls, pnode, pred_all, y_te):
    """全局排序口径：全部 vnode 距离算一遍，按类去重取最近一类，由宿主节点作答。
    同时按分解式计数：S_exp 为送对时本地 head 判对的条件概率，eps 为送错却答对的比例。"""
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
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    H_all, y_all, T, y_te = load_c100()
    if os.environ.get("NORM"):
        H_all = H_all / (np.linalg.norm(H_all, axis=1, keepdims=True) + 1e-12)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    nc, tr_sel, tr_node = build_split(y_all)
    H = H_all[tr_sel]
    y_tr = y_all[tr_sel]
    sizes = np.bincount(tr_node, minlength=N_NODES)
    log(f"训练 {H.shape} 测试 {T.shape} 类数 {len(np.unique(y_tr))}")
    log(f"节点 {N_NODES} 每节点 {PER_NODE_CLASSES} 类  "
        f"每节点样本 {sizes.min()}–{sizes.max()}")
    rec_per = {}
    for c in range(K):
        rec_per[c] = int(((y_tr == c)).sum() // 4)
    log(f"每类样本 {min(rec_per.values())}–{max(rec_per.values())}（4 个副本均分）")

    # 集中式参照之一：闭式 ridge，无随机性
    Wr = ridge_probe(H, y_tr, K)
    Tr = np.hstack([T.astype(np.float64), np.ones((len(T), 1))])
    accR = float(np.mean((Tr @ Wr).argmax(1) == y_te))
    log(f"集中式 ridge（闭式）                        {accR:.4f}")

    rec = {"A": [], "B": [], "C": [], "sys": []}
    aux = {"hit1": [], "recall3": [], "sexp": [], "eps": [], "n_vnode": []}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # A 同配方集中式：一个线性头，全部数据
        mA = train_linear(H, y_tr, K, seed=seed)
        accA = float(np.mean(predict(mA, T).argmax(1) == y_te))

        # B 纯本地：每节点一个 head，只在本地 2 类上训练
        heads = {}
        for n in range(N_NODES):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
        pred_all = {n: np.array(nc[n])[predict(heads[n], T).argmax(1)]
                    for n in range(N_NODES)}
        accB = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(N_NODES)]))

        # C FedAvg：服务器逐轮聚合，输出维对齐到 K
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

        # 本文：判别投影 d=K-1，每 (节点,类) 一个原型
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
            f"本文 召回@3={r3:.4f} 命中@1={h1:.4f} 端到端={e2e:.4f} | "
            f"S_exp={sexp:.4f} eps={eps:.4f} d_eff={d_eff}")
        del mA, heads, pred_all, glob
        if seed == SEEDS[0]:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out = {"dataset": "cifar100", "K": K, "n_nodes": N_NODES,
           "per_node_classes": PER_NODE_CLASSES, "replicas": N_NODES * PER_NODE_CLASSES // K,
           "d": K - 1, "proj": "lda", "n_test": N_TEST, "seeds": list(SEEDS),
           "ridge": accR, "acc": {}, "route": {}}

    log("")
    for key, name in (("A", "同配方集中式"), ("B", "纯本地"), ("C", "FedAvg"),
                      ("sys", "本文（k=1）")):
        v = np.array(rec[key])
        out["acc"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                               lo=float(v.min()), hi=float(v.max()))
        log(f"{name}: {v.mean():.4f} ± {v.std(ddof=1):.4f}  极差 {v.max() - v.min():.4f}")
    log(f"ridge（闭式，无种子）: {accR:.4f}")
    for key in ("recall3", "hit1", "sexp", "eps", "n_vnode"):
        v = np.array(aux[key], dtype=float)
        out["route"][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
        log(f"{key}: {v.mean():.4f} ± {v.std(ddof=1):.4f}")

    b = np.array(rec["sys"])
    for arr, name in ((np.array(rec["A"]), "同配方集中式"),
                      (np.array(rec["C"]), "FedAvg")):
        dif = (b - arr) * 100
        log(f"本文 − {name}: 均值 {dif.mean():+.2f} 点，逐种子 "
            + ", ".join(f"{x:+.2f}" for x in dif))
    dif = (b - accR) * 100
    log(f"本文 − ridge: 均值 {dif.mean():+.2f} 点，逐种子 "
        + ", ".join(f"{x:+.2f}" for x in dif))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
