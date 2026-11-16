# -*- coding: utf-8 -*-
"""exp92：只测自评这一步判断得对不对。

不做端到端、不测通信、不测 S_loc。只把每一次判断归到四个格子里的一个：

              实际送走        实际留在本地
  该送走       命中 (TP)       漏放 (FN)
  该留在本地   误送 (FP)       命中 (TN)

“该送走”的定义取本地答不对，即真类不在本地，或真类在本地但判别头答错。
另一条更宽的定义只取“真类不在本地”，一并算出放在 strict 字段，供对照。

判据取正文第 4.3 节那条：坐标到本地最近类原型的距离，分位点两折标定。
设定与第 5.1 节 100 类档相同：CIFAR-100、100 类、200 节点各 2 类、每类 4 副本。
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
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "1,2,3").split(","))
GRID = np.round(np.arange(0.0, 0.501, 0.01), 3)

OUT = os.path.join(ROOT, "results", f"results_selfcheck_decision_{N_TEST}.json")

COMP = [("A_uniform", None), ("R02", 0.02), ("R05", 0.05), ("R10", 0.10),
        ("R20", 0.20), ("R30", 0.30), ("R50", 0.50)]
NAME = {"A_uniform": "均匀(本地2%)", "R02": "本地98%", "R05": "本地95%",
        "R10": "本地90%", "R20": "本地80%", "R30": "本地70%",
        "R50": "本地50%"}


def log(*a):
    print(*a, flush=True)


def tab(w, FWD, SHOULD):
    """四格计数（加权）与由此推出的比率。"""
    tp = float((w * FWD * SHOULD).sum())
    fp = float((w * FWD * ~SHOULD).sum())
    fn = float((w * ~FWD * SHOULD).sum())
    tn = float((w * ~FWD * ~SHOULD).sum())
    tot = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                fwd_rate=(tp + fp) / tot,
                prec=prec, rec=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                miss=fn / max(tp + fn, 1e-12),          # 漏放率 = 1 - 查全
                false_fwd=fp / max(fp + tn, 1e-12),     # 误送率
                acc=(tp + tn) / tot)


def main():
    from exp32_end2end_all_lda import train_linear, predict
    from exp53_c100_end2end_lda import load_c100
    from exp91_selfcheck_c100 import (build_split, pct_two_fold, weights, auroc)

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

    out = {"setting": dict(K=K, n_nodes=Nn, per_node=2, n_test=int(len(T))),
           "seeds": {}, "mean": {}}

    acc_all = {}
    for seed in SEEDS:
        np.random.seed(seed)
        pred_all = {}
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            h = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
            pred_all[n] = np.array(nc[n])[predict(h, T).argmax(1)]

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = np.stack([pred_all[n] == y_te for n in range(Nn)]) & ID
        SHOULD = ~CORR            # 本地答不对，就该送走
        STRICT = ~ID              # 更宽的定义：真类不在本地

        P = np.zeros((Nn, len(T)), np.float32)
        for n in range(Nn):
            pr = np.stack([H[(tr_node == n) & (y_tr == c)].mean(0) for c in nc[n]])
            a = (T ** 2).sum(1)[:, None] + (pr ** 2).sum(1)[None, :] - 2 * T @ pr.T
            P[n] = -np.sqrt(np.maximum(a.min(1), 0))
        Pq = pct_two_fold(P, ID)

        idv = np.concatenate([Pq[n][ID[n]] for n in range(Nn)])
        odv = np.concatenate([Pq[n][~ID[n]] for n in range(Nn)])
        au = auroc(idv, odv)
        log(f"\n种子 {seed}: 该送走占 {SHOULD.mean():.3f}（strict {STRICT.mean():.3f}）"
            f" | 判据区分能力 {au:.4f}")

        rec = {"auroc": float(au), "should_rate": float(SHOULD.mean()),
               "strict_rate": float(STRICT.mean()), "comps": {}}
        for tag, r in COMP:
            w = weights(ID, r)
            rows = []
            for t in GRID:
                FWD = Pq < t
                d = tab(w, FWD, SHOULD)
                d["tau"] = float(t)
                d["strict"] = tab(w, FWD, STRICT)
                rows.append(d)
            rec["comps"][tag] = rows
            acc_all.setdefault(tag, []).append(rows)
        out["seeds"][str(seed)] = rec

    # ---- 汇总：按种子平均 ----
    log("\n" + "=" * 96)
    log("各构成：不同分位点下判断的对错")
    log("=" * 96)
    for tag, _ in COMP:
        S = acc_all[tag]
        r0 = S[0][0]
        base = (r0["tp"] + r0["fn"]) / (r0["tp"] + r0["fp"] + r0["fn"] + r0["tn"])
        log(f"\n【{NAME[tag]}】查询中该送走的占 {base:.3f}"
            f"（每个节点等权，合计 {r0['tp'] + r0['fp'] + r0['fn'] + r0['tn']:.0f}）")
        log(f"  全送走的平凡基准：查准 {base:.3f} 查全 1.000 准确率 {base:.3f}")
        log(f"{'分位点':>6}{'送走率':>8}{'查准':>7}{'查全':>7}{'F1':>7}"
            f"{'漏放':>7}{'误送':>7}{'准确率':>8}{'TP':>10}{'FN':>10}")
        best_f1 = None
        for k in (0, 2, 5, 10, 20, 30, 40, 50):
            if k >= len(S[0]):
                continue
            v = [s[k] for s in S]
            m = {kk: float(np.mean([x[kk] for x in v])) for kk in
                 ("tau", "fwd_rate", "prec", "rec", "f1", "miss",
                  "false_fwd", "acc", "tp", "fn")}
            star = ""
            if best_f1 is None or m["f1"] > best_f1[1]:
                best_f1 = (m["tau"], m["f1"])
            log(f"{m['tau']:>6.2f}{m['fwd_rate']:>8.3f}{m['prec']:>7.3f}"
                f"{m['rec']:>7.3f}{m['f1']:>7.3f}{m['miss']:>7.3f}"
                f"{m['false_fwd']:>7.3f}{m['acc']:>8.3f}"
                f"{m['tp']:>10.0f}{m['fn']:>10.0f}{star}")
        # 最优 F1 与最优准确率
        f1s = [float(np.mean([s[k]["f1"] for s in S])) for k in range(len(S[0]))]
        acs = [float(np.mean([s[k]["acc"] for s in S])) for k in range(len(S[0]))]
        kf, ka = int(np.argmax(f1s)), int(np.argmax(acs))
        log(f"  F1 最高：分位点 {GRID[kf]:.2f}，F1 {f1s[kf]:.3f}，"
            f"查准 {np.mean([s[kf]['prec'] for s in S]):.3f}，"
            f"查全 {np.mean([s[kf]['rec'] for s in S]):.3f}")
        log(f"  准确率最高：分位点 {GRID[ka]:.2f}，准确率 {acs[ka]:.3f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\n已写 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
