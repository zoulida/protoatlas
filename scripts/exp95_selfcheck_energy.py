# -*- coding: utf-8 -*-
"""exp95：把第 4.3 节的自评判据由“到最近类原型的距离”换成“能量”。

作者的裁定：能量 E=-log sum exp(f) 叙述更简单，一个标量、一个阈值，
不必存原型、不必算距离。本实验按 exp91 的全部口径重跑第 5.4 节的读数，
唯一改动是判据得分由 -距离 换成 能量取负（即 logsumexp）。

判据得分一律构造为“越大越像本地”：
    exp91  P = -d_n                          （越高越像本地）
    exp95  P = logsumexp(f) = -E_n           （能量越低越像本地）
阈值仍按本地样本的分位点两折标定，P >= t 即留在本地作答。

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
N_TEST = int(os.environ.get("NTEST", "10000"))
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "1,2,3").split(","))
GRID = np.round(np.arange(0.0, 0.401, 0.01), 3)

OUT = os.path.join(ROOT, "results", f"results_selfcheck_energy_{N_TEST}.json")


def log(*a):
    print(*a, flush=True)


def build_split(ytr):
    from exp1_route import make_shard2
    n_nodes = 2 * K
    nc, ni = make_shard2(ytr, n_nodes, K, 2, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni):
        for c in nc[n]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    return nc, np.concatenate(tr_sel), np.concatenate(tr_node)


def auroc(si, so):
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
    """分位点在本地样本内两折标定，外来样本用全本地表。score 越高越像本地。"""
    n, N = score.shape
    P = np.zeros_like(score)
    for i in range(n):
        ref = np.where(idmask[i])[0]
        A, B = ref[::2], ref[1::2]
        tA, tB, tAll = (np.sort(score[i][A]), np.sort(score[i][B]),
                        np.sort(score[i][ref]))
        P[i][B] = np.searchsorted(tA, score[i][B]) / len(tA)
        P[i][A] = np.searchsorted(tB, score[i][A]) / len(tB)
        m = np.ones(N, bool)
        m[ref] = False
        P[i][m] = np.searchsorted(tAll, score[i][m]) / len(tAll)
    return P


def route_ok(Zq, Zp, pcls, pnode, pred_all, y_te):
    """全局排序口径：按类去重取最近一类，由宿主节点作答。返回逐查询是否答对。"""
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2.0 * Zq @ Zp.T, 0))
    o = np.argsort(D, 1)
    ok = np.zeros(len(Zq), bool)
    hit = 0
    for i in range(len(Zq)):
        seen, top = set(), None
        for j in o[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.add(c)
            top = j
            break
        hit += int(pcls[top] == y_te[i])
        ok[i] = pred_all[int(pnode[top])][i] == y_te[i]
    return ok, hit / len(Zq)


def weights(ID, r):
    n, N = ID.shape
    if r is None:
        return np.ones((n, N), np.float64)
    w = np.zeros((n, N), np.float64)
    for i in range(n):
        idn = np.where(ID[i])[0]
        odn = np.where(~ID[i])[0]
        w[i, idn] = (1 - r) / len(idn)
        w[i, odn] = r / len(odn)
    return w


def sweep(P, ID, CORR, ROK, w):
    tot = w.sum()
    wid, wod = (w * ID).sum(), (w * ~ID).sum()
    rows = []
    for t in GRID:
        acc = P >= t
        wa = (w * acc).sum()
        rows.append(dict(
            tau=float(t), eta=float(wa / tot),
            sloc=float((w * acc * CORR).sum() / max(wa, 1e-9)),
            e2e=float(((w * acc * CORR).sum()
                       + (w * (~acc) * ROK[None, :]).sum()) / tot),
            leak=float((w * acc * ~ID).sum() / max(wod, 1e-9)),
            miss=float((w * (~acc) * ID * CORR).sum() / max(wid, 1e-9)),
            misfwd=float((w * (~acc) * ID).sum() / max(wid, 1e-9)),
        ))
    return rows


def main():
    from exp32_end2end_all_lda import train_linear, predict
    from exp53_c100_end2end_lda import load_c100

    H_all, y_all, _, _ = load_c100()
    m_tr = y_all < K
    H_all, y_all = H_all[m_tr], y_all[m_tr]
    T_all = np.load(os.path.join(CACHE, "feat_c100_test10k.npy")).astype(np.float32)
    y_all_te = np.load(os.path.join(CACHE, "c100_test10k.npz"))["y"].astype(np.int64)

    rng = np.random.RandomState(0)
    per = N_TEST // K
    idx_te = np.concatenate([
        rng.choice(np.where(y_all_te == c)[0], size=min(per, (y_all_te == c).sum()),
                   replace=False) for c in range(K)])
    T, y_te = T_all[idx_te], y_all_te[idx_te]

    nc, tr_sel, tr_node = build_split(y_all)
    H, y_tr = H_all[tr_sel], y_all[tr_sel]
    Nn = 2 * K
    log(f"K={K} 节点 {Nn} 每节点 {len(nc[0])} 类 训练 {H.shape} 查询 {T.shape}")

    out = {"setting": dict(K=K, n_nodes=Nn, per_node=2, n_test=int(len(T)),
                           criterion="energy: -log sum exp(f)"),
           "seeds": {}, "summary": {}}

    for seed in SEEDS:
        np.random.seed(seed)
        heads, pred_all = {}, {}
        P = np.zeros((Nn, len(T)), np.float32)
        PRED = np.zeros((Nn, len(T)), np.int64)
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
            lg = predict(heads[n], T)
            PRED[n] = np.array(nc[n])[lg.argmax(1)]
            pred_all[n] = PRED[n]
            mx = lg.max(1, keepdims=True)
            P[n] = (np.log(np.exp(lg - mx).sum(1)) + mx[:, 0]).astype(np.float32)

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = (PRED == y_te[None, :]) & ID
        WRONG = ID & ~CORR

        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in nc[n]:
                protos.append(H[(tr_node == n) & (y_tr == c)].mean(0))
                pcls.append(c)
                pnode.append(n)
        Zp = np.stack(protos).astype(np.float32)
        ROK, hit1 = route_ok(T, Zp, np.array(pcls), np.array(pnode), pred_all, y_te)
        log(f"  种子 {seed}: 路由命中 {hit1:.4f} 送走后答对 {ROK.mean():.4f} "
            f"本地命中 {ID.mean():.3f}")

        Pq = pct_two_fold(P, ID)
        au = auroc(P[ID], P[~ID])
        au_err = auroc(P[ID & CORR], P[WRONG])
        sid = float(CORR.sum() / ID.sum())
        rec = {"route_hit1": float(hit1), "forward_ok": float(ROK.mean()),
               "local_hit": float(ID.mean()), "auroc": float(au),
               "auroc_err": float(au_err), "s_id": sid, "comps": {}}
        log(f"  AUROC 本地/外来 {au:.4f} | 答对/答错 {au_err:.4f} | "
            f"本地类上的判别精度 {sid:.4f}")

        COMP = [("A_uniform", None), ("R02", 0.02), ("R05", 0.05), ("R10", 0.10),
                ("R20", 0.20), ("R30", 0.30), ("R50", 0.50)]
        for tag, r in COMP:
            w = weights(ID, r)
            rows = sweep(Pq, ID, CORR, ROK, w)
            allLocal = float((w * CORR).sum() / w.sum())
            allFwd = float((w * ROK[None, :]).sum() / w.sum())
            best = max(rows, key=lambda x: x["e2e"])
            ub = (1 - r) * sid + r * float(ROK.mean()) if r is not None \
                else float(ID.mean() * sid + (1 - ID.mean()) * ROK.mean())
            rec["comps"][tag] = dict(rows=rows, best=best, upper=ub,
                                     all_local=allLocal, all_forward=allFwd)
            keep = (0.05, 0.10, 0.20, 0.30)
            at = [x for x in rows if any(abs(x["tau"] - t) < 1e-6 for t in keep)]
            log(f"    {tag}: 全本地 {allLocal:.4f} 全送走 {allFwd:.4f} 上限 {ub:.4f} | "
                f"最优 tau={best['tau']:.2f} E2E={best['e2e']:.4f} eta={best['eta']:.3f} "
                f"S_loc={best['sloc']:.4f} 漏放={best['leak']:.3f}")
            log("      " + "  ".join(
                f"tau{x['tau']:.2f}:E2E{x['e2e']:.4f}/eta{x['eta']:.2f}/漏{x['leak']:.2f}"
                for x in at))
        out["seeds"][str(seed)] = rec

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"已写 {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
