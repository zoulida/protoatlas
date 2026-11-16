# -*- coding: utf-8 -*-
"""exp94b：在相同送走率下比较各种判据（修正版）。

修正 exp94 的两处错误：
  1. np.minimum(Pd,Pm) < t 是“任一判送走”即逻辑或，不是“与”。
     严格版应为 np.maximum(Pd,Pm) < t。
  2. exp93 里外来样本的近邻纯度没有计算，被初始化成 0，
     导致“knn 在本地 vs 外来上 0.9996”是人为构造的假值。

评价改为固定实际送走率（0.02/0.05/0.10/0.20），比较：
  漏放（总）、漏放中的外来部分、漏放中的本地答错部分、
  以及按代价加权的损失（误送丢 0.19，漏放丢 0.80）。

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
QS = (0.02, 0.05, 0.10, 0.20)
OUT = os.path.join(ROOT, "results", f"results_local_confidence_cmp_{N_TEST}.json")

COMP = [("A_uniform", None), ("R05", 0.05), ("R20", 0.20), ("R50", 0.50)]
NAME = {"A_uniform": "均匀(本地2%)", "R05": "本地95%", "R20": "本地80%",
        "R50": "本地50%"}


def log(*a):
    print(*a, flush=True)


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

    out = {"setting": dict(K=K, n_nodes=Nn, per_node=2, n_test=int(len(T)), knn=KNN),
           "seeds": {}}
    store = {}

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

            # 近邻纯度：全部查询样本都要算
            ref = H[tr_node == n]
            refy = y_tr[tr_node == n]
            a2 = (T ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None, :] - 2 * T @ ref.T
            nb = np.argpartition(a2, KNN, axis=1)[:, :KNN]
            KN[n] = (refy[nb] == PRED[n][:, None]).mean(1)

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = np.stack([PRED[n] == y_te for n in range(Nn)]) & ID
        SHOULD = ~CORR
        WRONG_LOCAL = ID & ~CORR            # 真类在本地但答错
        FOREIGN = ~ID                        # 真类不在本地

        Pd = pct_two_fold(-D, ID)
        Pm = pct_two_fold(MARGIN, ID)
        Pk = pct_two_fold(KN, ID)
        SC = {"d": Pd, "margin": Pm, "knn": Pk,
              "strict": np.maximum(Pd, Pm),      # 两个都要过才留本地
              "loose": np.minimum(Pd, Pm),       # 任一个过就留本地
              "avg": 0.5 * Pd + 0.5 * Pm,
              "d+knn": np.maximum(Pd, Pk)}

        log(f"\n种子 {seed}: ID {int(ID.sum())} 答错 {int(WRONG_LOCAL.sum())}"
            f" 外来 {int(FOREIGN.sum())} | 该送走 {SHOULD.mean():.4f}")
        for k, v in SC.items():
            a1 = float(np.mean([__import__("exp93_local_confidence").auroc_rank(
                v[n][ID[n] & CORR[n]], v[n][ID[n] & ~CORR[n]])
                for n in range(Nn) if (ID[n] & ~CORR[n]).sum() > 3]))
            a2 = float(np.mean([__import__("exp93_local_confidence").auroc_rank(
                v[n][ID[n]], v[n][~ID[n]]) for n in range(Nn)]))
            log(f"  {k:<7} 答对vs答错 {a1:.4f}   本地vs外来 {a2:.4f}")
            store.setdefault((k, "wle"), []).append(a1)
            store.setdefault((k, "lvs"), []).append(a2)

        for tag, r in COMP:
            w = weights(ID, r)
            for s in SC:
                for q in QS:
                    FWD = SC[s] < q
                    tp = float((w * FWD * SHOULD).sum())
                    fp = float((w * FWD * ~SHOULD).sum())
                    fn = float((w * ~FWD * SHOULD).sum())
                    tn = float((w * ~FWD * ~SHOULD).sum())
                    tot = tp + fp + fn + tn
                    fn_foreign = float((w * ~FWD * FOREIGN).sum())
                    fn_wrong = float((w * ~FWD * WRONG_LOCAL).sum())
                    loss = fn / tot * 0.80 + fp / tot * 0.19
                    store.setdefault((s, tag, q), []).append(
                        dict(tot=tot, tp=tp, fp=fp, fn=fn, tn=tn,
                             fwd=(tp + fp) / tot, miss=fn / max(tp + fn, 1e-12),
                             miss_f=fn_foreign / max(float((w * FOREIGN).sum()), 1e-9),
                             miss_w=fn_wrong / max(float((w * WRONG_LOCAL).sum()), 1e-9),
                             acc=(tp + tn) / tot, loss=loss,
                             base=(tp + fn) / tot))
        out["seeds"][str(seed)] = {"id": int(ID.sum()),
                                   "wrong": int(WRONG_LOCAL.sum())}

    # ---- 汇总 ----
    log("\n" + "=" * 104)
    log("同一送走率下各判据的漏放（多种子均值，本地命中 95% 一档）")
    log("=" * 104)
    tag = "R05"
    for q in QS:
        log(f"\n送走率 {q:.2f}")
        log(f"{'判据':<8}{'实际送走':>9}{'漏放':>8}{'外来漏放':>9}"
            f"{'本地答错漏放':>13}{'准确率':>8}{'加权损失':>10}")
        for s in ("d", "margin", "knn", "strict", "loose", "avg", "d+knn"):
            v = store[(s, tag, q)]
            m = {k: float(np.mean([x[k] for x in v])) for k in
                 ("fwd", "miss", "miss_f", "miss_w", "acc", "loss")}
            log(f"{s:<8}{m['fwd']:>9.3f}{m['miss']:>8.3f}{m['miss_f']:>9.3f}"
                f"{m['miss_w']:>13.3f}{m['acc']:>8.3f}{m['loss']:>10.4f}")

    log("\n" + "=" * 104)
    log("其他构成，送走率 0.10，加权损失")
    log("=" * 104)
    for tag, _ in COMP:
        log(f"\n【{NAME[tag]}】")
        log(f"{'判据':<8}{'漏放':>8}{'误送':>8}{'准确率':>8}{'加权损失':>10}")
        for s in ("d", "margin", "knn", "strict", "loose", "avg", "d+knn"):
            v = store[(s, tag, 0.10)]
            m = {k: float(np.mean([x[k] for x in v])) for k in
                 ("miss", "acc", "loss")}
            fp_rate = float(np.mean([x["fp"] / max(x["fp"] + x["tn"], 1e-9) for x in v]))
            log(f"{s:<8}{m['miss']:>8.3f}{fp_rate:>8.3f}{m['acc']:>8.3f}"
                f"{m['loss']:>10.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\n已写 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
