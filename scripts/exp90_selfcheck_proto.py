"""exp90: 自评一定要用判别头的分值吗？——用"到本地类原型的距离"作判据

两条路的差别：
  能量   : 需要判别头（本地训练，各节点尺度不同，须逐节点标定）
  原型距离: 只需要 vnode 坐标（本就要注册），与路由用的是同一把尺子，
            坐标空间全网统一，尺度不由本地训练决定

本实验把两条判据放在同一批样本、同一套标定上比：
  AUROC（本地 vs 外来）、3%/5% 求助构成下的最优阈值与端到端。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def auroc(si, so):
    s = np.concatenate([si, so])
    y = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    r = np.empty(len(s), np.float64)
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct_two_fold(score, idmask):
    """只在本地样本内分折标定，外来样本用全本地表"""
    n, N = score.shape
    P = np.zeros_like(score)
    for i in range(n):
        ref = np.where(idmask[i])[0]
        A, B = ref[::2], ref[1::2]
        tA, tB, tAll = np.sort(score[i][A]), np.sort(score[i][B]), np.sort(score[i][ref])
        P[i][B] = np.searchsorted(tA, score[i][B]) / len(tA)
        P[i][A] = np.searchsorted(tB, score[i][A]) / len(tB)
        m = np.ones(N, bool)
        m[ref] = False
        P[i][m] = np.searchsorted(tAll, score[i][m]) / len(tAll)
    return P


def sweep(P, ID, CORR, y_te, r):
    n, N = P.shape
    w = np.zeros_like(P, dtype=np.float64)
    for i in range(n):
        idn = np.where(ID[i])[0]
        odn = np.where(~ID[i])[0]
        w[i, idn] = (1 - r) / len(idn)
        w[i, odn] = r / len(odn)
    tot = w.sum()
    best, rows = None, []
    for t in np.round(np.arange(0.0, 0.96, 0.01), 2):
        defer = P < t                 # 分位越低越不像本地，送走
        loc = ~defer
        right_local = (w * loc * CORR).sum() / tot
        fwd = (w * defer).sum() / tot
        leak = (w * loc * ~ID).sum() / max((w * ~ID).sum(), 1)
        misfwd = (w * defer * ID).sum() / max((w * ID).sum(), 1)
        e2e = right_local + 0.94 * fwd
        rows.append(dict(tau=float(t), fwd=float(fwd), leak=float(leak),
                         misfwd=float(misfwd), e2e=float(e2e)))
        if best is None or e2e > best["e2e"]:
            best = rows[-1]
    return rows, best


def main():
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    Nn = C["n_nodes"]
    cls_of = {i: sorted(nc[i]) for i in range(Nn)}

    # 原型：每节点每类一个类均值（= vnode 坐标）
    Z = T                                        # 本文坐标即特征本身
    D = np.zeros((Nn, len(T)), np.float32)
    for i in range(Nn):
        pr = np.stack([H[(tr_node == i) & (y_tr == k)].mean(0) for k in cls_of[i]])
        a = (Z ** 2).sum(1)[:, None] + (pr ** 2).sum(1)[None, :] - 2 * Z @ pr.T
        D[i] = np.sqrt(np.maximum(a.min(1), 0))
    SC_proto = -D                                 # 越高越像本地

    L = np.load(os.path.join(CACHE, "tmp_L88.npy"))   # 能量（logsumexp），越高越像本地
    ID = np.load(os.path.join(CACHE, "tmp_ID88.npy"))
    PRED = np.load(os.path.join(CACHE, "tmp_PRED88.npy"))
    CORR = (PRED == y_te[None, :]) & ID

    res = {}
    for name, S in [("能量(判别头)", L), ("原型距离", SC_proto)]:
        idv = np.concatenate([S[i][ID[i]] for i in range(Nn)])
        odv = np.concatenate([S[i][~ID[i]] for i in range(Nn)])
        a = auroc(idv, odv)
        P = pct_two_fold(S, ID)
        res[name] = {"auroc": a}
        print(f"\n=== {name} ===  AUROC {a:.4f}")
        for r in (0.03, 0.05):
            rows, best = sweep(P, ID, CORR, y_te, r)
            res[name][f"r={r}"] = best
            at5 = min(rows, key=lambda x: abs(x["tau"] - 0.05))
            at10 = min(rows, key=lambda x: abs(x["tau"] - 0.10))
            print(f"  求助 {r:.0%}：最优 τ={best['tau']:.2f} E2E={best['e2e']:.4f} "
                  f"转出 {best['fwd']:.3f} 漏放 {best['leak']:.3f} 误送 {best['misfwd']:.3f}")
            print(f"        τ=0.05 E2E={at5['e2e']:.4f} 漏放 {at5['leak']:.3f}；"
                  f"τ=0.10 E2E={at10['e2e']:.4f} 漏放 {at10['leak']:.3f}")

    with open(os.path.join(ROOT, "results", "results_selfcheck_proto.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写 results_selfcheck_proto.json")


if __name__ == "__main__":
    main()
