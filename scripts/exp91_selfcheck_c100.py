# -*- coding: utf-8 -*-
"""exp91：100 类主设定下，自评一步的读数。

与 exp90 的区别：
  exp90 在 CIFAR-10（10 类、20 节点各 2 类）上比三条判据；
  本实验回到正文第 5 章的主设定（CIFAR-100、100 类、200 节点各 2 类、
  每类 4 副本），只取正文 4.3 用的一条判据，即坐标到本地最近类原型的距离。

两个问题：
  (1) 两种查询构成下，自评的本地解决率 eta、本地作答精度 S_loc、漏放率各是多少；
  (2) 端到端精度对阈值分位点 tau 是否敏感，最优位置在哪。

查询构成：
  A 均匀：查询在 100 类上均匀抽样，每节点本地命中 2%（即第 5.1 节口径）；
  B 业务：每节点收到的查询里 95% 落在自己持有的类上，5% 需要求助。
端到端按附录 C 的分解逐样本累加，不取常数近似：
  E2E = P(本地作答且答对) + P(送走且目标答对)
其中“送走答对”由同一套划分上跑一遍全局排序路由实测得到。
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

OUT = os.path.join(ROOT, "results", f"results_selfcheck_c100_{N_TEST}.json")


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
    r = np.empty(len(s), np.float64)
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
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
    """r=None 为自然构成（均权）；否则 r 为该节点收到的查询中求助所占比例"""
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

    out = {"setting": dict(K=K, n_nodes=Nn, per_node=2, n_test=int(len(T))),
           "seeds": {}, "summary": {}}

    for seed in SEEDS:
        np.random.seed(seed)
        heads, pred_all = {}, {}
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([nc[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(nc[n]), seed=seed * 100 + n)
            pred_all[n] = np.array(nc[n])[predict(heads[n], T).argmax(1)]

        ID = np.stack([np.isin(y_te, nc[n]) for n in range(Nn)])
        CORR = np.stack([pred_all[n] == y_te for n in range(Nn)]) & ID

        P = np.zeros((Nn, len(T)), np.float32)
        for n in range(Nn):
            pr = np.stack([H[(tr_node == n) & (y_tr == c)].mean(0) for c in nc[n]])
            a = (T ** 2).sum(1)[:, None] + (pr ** 2).sum(1)[None, :] - 2 * T @ pr.T
            P[n] = -np.sqrt(np.maximum(a.min(1), 0))      # 取负距离，越高越像本地
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
        idv = np.concatenate([P[n][ID[n]] for n in range(Nn)])
        odv = np.concatenate([P[n][~ID[n]] for n in range(Nn)])
        au = auroc(idv, odv)
        sid = float(CORR.sum() / ID.sum())
        rec = {"route_hit1": float(hit1), "forward_ok": float(ROK.mean()),
               "local_hit": float(ID.mean()), "auroc": float(au),
               "s_id": sid, "comps": {}}
        log(f"  AUROC {au:.4f} | 本地类上的判别精度 {sid:.4f}")
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
