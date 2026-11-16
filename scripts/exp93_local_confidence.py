# -*- coding: utf-8 -*-
"""exp93：本地能不能判出“自己答得对不对”。

问题：第 4.3 节的判据 d_n 回答的是“这个输入像不像我持有的类”，
不回答“我答得对不对”。exp92 已实测出这个缺口：真类在本地而判别头
答错的样本，漏放 58.5%。

本实验在 ID 样本（真类在本地）上把二分改成“本地答对 / 本地答错”，
比较各类量把这个二分分开的能力（AUROC），并测两阶段（先挡外来、
再挡答错）能把四格改善多少。

量的定义（一律构造为“值越大越可信”）：
    neg_d       -d_n，到最近类原型距离取负（正文现用判据）
    neg_E       logsumexp(logits)，能量取负
    margin      top1 与 top2 的 logit 之差
    msp         softmax 最大值
    neg_H       负熵
    ratio       -d_n / sigma，sigma 为最近类的类内平均距离
    knn         本地 k 近邻中同属最近类的比例

设定同第 5.1 节 100 类档：CIFAR-100、200 节点各 2 类、每类 4 副本。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

K = 100
N_TEST = int(os.environ.get("NTEST", "10000"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "1").split(","))
KNN = 10
OUT = os.path.join(ROOT, "results", f"results_local_confidence_{N_TEST}.json")


def log(*a):
    print(*a, flush=True)


def auroc_rank(si, so):
    """并列取平均秩。"""
    s = np.concatenate([si, so])
    y = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    r = np.empty(len(s), np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    from exp32_end2end_all_lda import train_linear, predict
    from exp53_c100_end2end_lda import load_c100
    from exp91_selfcheck_c100 import build_split

    H_all, y_all, _, _ = load_c100()
    m_tr = y_all < K
    H_all, y_all = H_all[m_tr], y_all[m_tr]
    T_all = np.load(os.path.join(CACHE, "feat_c100_test10k.npy")).astype(np.float32)
    y_te_all = np.load(os.path.join(CACHE, "c100_test10k.npz"))["y"].astype(np.int64)

    rng = np.random.RandomState(0)
    per = N_TEST // K
    idx = np.concatenate([rng.choice(np.where(y_te_all == c)[0],
                                     size=min(per, (y_te_all == c).sum()),
                                     replace=False) for c in range(K)])
    T, y_te = T_all[idx], y_te_all[idx]

    nc, tr_sel, tr_node = build_split(y_all)
    H, y_tr = H_all[tr_sel], y_all[tr_sel]
    Nn = 2 * K
    log(f"K={K} 节点 {Nn} 训练 {H.shape} 查询 {T.shape} 种子 {SEEDS}")

    out = {"setting": dict(K=K, n_nodes=Nn, per_node=2, n_test=int(len(T)), knn=KNN),
           "seeds": {}, "mean": {}}
    acc = {}

    for seed in SEEDS:
        np.random.seed(seed)
        PRED = np.zeros((Nn, len(T)), np.int64)
        LG = np.zeros((Nn, len(T), 2), np.float32)
        D = np.zeros((Nn, len(T)), np.float32)       # 到最近原型的距离
        DC = np.zeros((Nn, len(T)), np.float32)      # 到最近类内样本的距离
        SIG = np.zeros(Nn, np.float32)               # 最近类的类内平均距离

        for n in range(Nn):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            h = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
            lg = predict(h, T)
            LG[n] = lg
            PRED[n] = np.array(nc[n])[lg.argmax(1)]

            pr = np.stack([H[(tr_node == n) & (y_tr == c)].mean(0) for c in nc[n]])
            a = (T ** 2).sum(1)[:, None] + (pr ** 2).sum(1)[None, :] - 2 * T @ pr.T
            dd = np.sqrt(np.maximum(a, 0))
            D[n] = dd.min(1)
            # 类内平均距离：各训练样本到本类原型的距离的均值
            s = [np.linalg.norm(H[(tr_node == n) & (y_tr == c)] - pr[i], axis=1).mean()
                 for i, c in enumerate(nc[n])]
            SIG[n] = float(np.mean(s))

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = np.stack([PRED[n] == y_te for n in range(Nn)]) & ID

        # 各类量
        mx = LG.max(2)
        lg_sum = np.log(np.exp(LG - mx[:, :, None]).sum(2)) + mx
        srt = np.sort(LG, 2)
        MARGIN = srt[:, :, -1] - srt[:, :, -2]
        p = np.exp(LG - mx[:, :, None])
        p /= p.sum(2, keepdims=True)
        MSP = p.max(2)
        ENT = -(p * np.log(np.maximum(p, 1e-12))).sum(2)

        FEATS = {
            "neg_d": -D,
            "neg_E": lg_sum,
            "margin": MARGIN,
            "msp": MSP,
            "neg_H": -ENT,
            "ratio": -D / SIG[:, None],
        }

        # kNN 纯度：只在 ID 样本上算，参照为本地训练样本
        KN = np.zeros((Nn, len(T)), np.float32)
        for n in range(Nn):
            ii = np.where(ID[n])[0]
            if len(ii) == 0:
                continue
            ref = H[tr_node == n]
            refy = y_tr[tr_node == n]
            a = (T[ii] ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * T[ii] @ ref.T
            nb = np.argsort(a, axis=1)[:, :KNN]
            # 与预测类相同者占的比例
            KN[n, ii] = (refy[nb] == PRED[n][ii][:, None]).mean(1)
        FEATS["knn"] = KN

        n_id = int(ID.sum())
        n_err = int((ID & ~CORR).sum())
        log(f"\n种子 {seed}: ID 样本 {n_id}，其中判错 {n_err}（{n_err / max(n_id, 1):.4f}）")

        rec = {"n_id": n_id, "n_err": n_err, "auroc": {}}
        log(f"{'量':<8}{'答对 vs 答错':>14}{'本地 vs 外来':>14}")
        for k, v in FEATS.items():
            a1 = auroc_rank(v[ID & CORR], v[ID & ~CORR])
            a2 = auroc_rank(v[ID], v[~ID])
            rec["auroc"][k] = dict(err=float(a1), idout=float(a2))
            acc.setdefault(k, []).append(a1)
            log(f"{k:<8}{a1:>14.4f}{a2:>14.4f}")

        out["seeds"][str(seed)] = rec

    log("\n" + "=" * 60)
    log("跨种子均值：答对 vs 答错 的区分能力")
    for k, v in sorted(acc.items(), key=lambda x: -np.mean(x[1])):
        log(f"  {k:<8} {np.mean(v):.4f}")
    out["mean"] = {k: float(np.mean(v)) for k, v in acc.items()}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\n已写 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
