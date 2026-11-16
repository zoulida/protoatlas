# -*- coding: utf-8 -*-
"""exp94c：判据比较的最终版——各自取最优操作点。

exp94b 的问题：用分位点 q 当“送走率”比较，但 loose（min）在 q=0.10 时
实际送走 0.195、knn 只有 0.089，并不对齐。本版改为对每个判据扫遍所有
分位点，各自取加权损失最小的一点再比。

代价口径：误送一个本可答对的样本丢 0.19（本地 0.9926 对送走链路 0.8023），
漏放一个该送走的丢 0.80（本地必错，送走能对 0.8023）。

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
N_TEST = int(os.environ.get("NTEST", "3000"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "1").split(","))
KNN = 10
GRID = np.round(np.arange(0.0, 0.501, 0.01), 3)
OUT = os.path.join(ROOT, "results", f"results_local_confidence_final_{N_TEST}.json")

COMP = [("A_uniform", None), ("R05", 0.05), ("R20", 0.20), ("R50", 0.50)]
NAME = {"A_uniform": "均匀(本地2%)", "R05": "本地95%", "R20": "本地80%",
        "R50": "本地50%"}
SCHEMES = ("d", "margin", "knn", "strict", "loose", "avg", "d+knn", "d+margin+knn")
C_FP, C_FN = 0.19, 0.80


def log(*a):
    print(*a, flush=True)


def auroc_rank(si, so):
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


def pct_two_fold(score, idmask):
    Nn, Nt = score.shape
    P = np.zeros_like(score, dtype=np.float32)
    for n in range(Nn):
        ref = np.where(idmask[n])[0]
        if len(ref) < 4:
            P[n] = 0.5
            continue
        A, B = ref[::2], ref[1::2]
        tA, tB = np.sort(score[n][A]), np.sort(score[n][B])
        tAll = np.sort(score[n][ref])
        P[n][B] = np.searchsorted(tA, score[n][B], side="right") / len(tA)
        P[n][A] = np.searchsorted(tB, score[n][A], side="right") / len(tB)
        m = np.ones(Nt, bool)
        m[ref] = False
        P[n][m] = np.searchsorted(tAll, score[n][m], side="right") / len(tAll)
    return P


def weights(ID, r):
    Nn, Nt = ID.shape
    w = np.zeros((Nn, Nt), np.float64)
    if r is None:
        w[:] = 1.0
        return w
    for n in range(Nn):
        ii, oo = ID[n], ~ID[n]
        if ii.sum():
            w[n][ii] = (1 - r) / ii.sum()
        if oo.sum():
            w[n][oo] = r / oo.sum()
    return w


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

    AU = {}
    BEST = {}
    out = {"setting": dict(K=K, n_nodes=Nn, n_test=int(len(T)), knn=KNN,
                           c_fp=C_FP, c_fn=C_FN), "seeds": {}}

    for seed in SEEDS:
        np.random.seed(seed)
        PRED = np.zeros((Nn, len(T)), np.int64)
        D = np.zeros((Nn, len(T)), np.float32)
        MARGIN = np.zeros((Nn, len(T)), np.float32)
        KN = np.zeros((Nn, len(T)), np.float32)

        for n in range(Nn):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            h = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
            lg = predict(h, T)
            PRED[n] = np.array(nc[n])[lg.argmax(1)]
            s = np.sort(lg, 1)
            MARGIN[n] = s[:, -1] - s[:, -2]

            pr = np.stack([H[(tr_node == n) & (y_tr == c)].mean(0) for c in nc[n]])
            a = (T ** 2).sum(1)[:, None] + (pr ** 2).sum(1)[None, :] - 2 * T @ pr.T
            D[n] = np.sqrt(np.maximum(a.min(1), 0))

            ref = H[tr_node == n]
            refy = y_tr[tr_node == n]
            a2 = (T ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * T @ ref.T
            nb = np.argpartition(a2, KNN, axis=1)[:, :KNN]
            KN[n] = (refy[nb] == PRED[n][:, None]).mean(1)

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = np.stack([PRED[n] == y_te for n in range(Nn)]) & ID
        SHOULD = ~CORR
        WRONG = ID & ~CORR

        Pd = pct_two_fold(-D, ID)
        Pm = pct_two_fold(MARGIN, ID)
        Pk = pct_two_fold(KN, ID)
        SC = {"d": Pd, "margin": Pm, "knn": Pk,
              "strict": np.maximum(Pd, Pm),
              "loose": np.minimum(Pd, Pm),
              "avg": 0.5 * Pd + 0.5 * Pm,
              "d+knn": np.maximum(Pd, Pk),
              "d+margin+knn": np.maximum(np.maximum(Pd, Pm), Pk)}

        log(f"\n种子 {seed}: ID {int(ID.sum())} 答错 {int(WRONG.sum())}"
            f" 外来 {int((~ID).sum())}")
        log(f"{'判据':<14}{'答对vs答错':>12}{'本地vs外来':>12}")
        for k, v in SC.items():
            a1 = auroc_rank(v[ID & CORR], v[WRONG])
            a2 = auroc_rank(v[ID], v[~ID])
            AU.setdefault(k, []).append((a1, a2))
            log(f"{k:<14}{a1:>12.4f}{a2:>12.4f}")

        for tag, r in COMP:
            w = weights(ID, r)
            ws, wf = w * SHOULD, w * ~SHOULD
            for s in SCHEMES:
                best = None
                for t in GRID:
                    FWD = SC[s] < t
                    fn = float((ws * ~FWD).sum())
                    fp = float((wf * FWD).sum())
                    tp = float((ws * FWD).sum())
                    tn = float((wf * ~FWD).sum())
                    tot = tp + fp + fn + tn
                    loss = fn / tot * C_FN + fp / tot * C_FP
                    if best is None or loss < best["loss"]:
                        best = dict(loss=loss, tau=float(t),
                                    fwd=(tp + fp) / tot,
                                    miss=fn / max(tp + fn, 1e-12),
                                    fp_rate=fp / max(fp + tn, 1e-12),
                                    acc=(tp + tn) / tot)
                BEST.setdefault((s, tag), []).append(best)
        out["seeds"][str(seed)] = dict(id=int(ID.sum()), wrong=int(WRONG.sum()))

    log("\n" + "=" * 110)
    log("各判据在**各自最优操作点**上的加权损失（误送 0.19、漏放 0.80）")
    log("=" * 110)
    for tag, _ in COMP:
        log(f"\n【{NAME[tag]}】")
        log(f"{'判据':<14}{'最优损失':>10}{'分位点':>8}{'送走率':>8}"
            f"{'漏放':>8}{'误送':>8}{'准确率':>8}")
        rows = []
        for s in SCHEMES:
            v = BEST[(s, tag)]
            m = {k: float(np.mean([x[k] for x in v])) for k in
                 ("loss", "tau", "fwd", "miss", "fp_rate", "acc")}
            rows.append((m["loss"], s, m))
        for loss, s, m in sorted(rows):
            log(f"{s:<14}{m['loss']:>10.4f}{m['tau']:>8.2f}{m['fwd']:>8.3f}"
                f"{m['miss']:>8.3f}{m['fp_rate']:>8.3f}{m['acc']:>8.3f}")

    log("\n" + "=" * 110)
    log("区分能力（多种子均值）")
    log(f"{'判据':<14}{'答对vs答错':>12}{'本地vs外来':>12}")
    for s, v in sorted(AU.items(), key=lambda x: -np.mean([a for a, _ in x[1]])):
        log(f"{s:<14}{np.mean([a for a, _ in v]):>12.4f}"
            f"{np.mean([b for _, b in v]):>12.4f}")

    out["auroc"] = {s: dict(err=float(np.mean([a for a, _ in v])),
                            idout=float(np.mean([b for _, b in v])))
                    for s, v in AU.items()}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\n已写 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
