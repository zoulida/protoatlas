"""exp88: 自评阈值该取多少 —— 扫阈值，报查准/查全

与前几轮的区别：
  exp20/21 只报 AUROC 与"5% 误拒"这一个点，5% 是随手定的，没有依据。
  本实验把阈值当自由变量扫一遍，给出查准率、查全率、F1 与代价，找出最优点。

口径（与 4.2 的叙述位置一致，判据设在**请求方**一侧）：
  节点 n 收到一个查询 x，用本地判别头算 logsumexp，再换算成"在本地未见样本
  分布上的百分位" p。p < τ 即判成答不出，转入路由；p >= τ 本地作答。
  p 由两折交叉得到：偶数折的百分位表标定奇数折，反之亦然。

两个"该拒"的定义：
  strict : 真类不在本节点持有的类里（答不了）
  err    : 真类不在本地 或 在本地但本地答错（答不了 + 答错）

查询构成两档：
  uniform : 每节点 10 类等概率（本地命中率 = 2/10 = 20%）
  skew50  : 每节点本地类占一半（本地命中率 = 50%）
"""
import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    Nn = C["n_nodes"]
    cls_of = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"设定：{Nn} 节点，每节点 {len(cls_of[0])} 类，测试 {len(T)} 张，"
        f"训练 {len(H)} 张")

    # ---- 每个节点的判别头，与 exp20 同参 ----
    heads = {}
    for n in range(Nn):
        m = tr_node == n
        idx = np.array([cls_of[n].index(c) for c in y_tr[m]])
        heads[n] = train_head(H[m], idx)

    # ---- 在全部测试样本上算 logsumexp 与预测 ----
    L = np.zeros((Nn, len(T)), np.float32)
    PRED = np.zeros((Nn, len(T)), np.int64)
    for n in range(Nn):
        arr = np.array(cls_of[n])
        with torch.no_grad():
            lg = heads[n](torch.from_numpy(T)).numpy()
        L[n] = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
        PRED[n] = arr[lg.argmax(1)]

    ID = np.stack([np.isin(y_te, cls_of[n]) for n in range(Nn)])   # (Nn, Nt)
    CORR = (PRED == y_te[None, :]) & ID

    # ---- 两折交叉标定百分位 ----
    P = np.zeros_like(L)
    for n in range(Nn):
        ref_all = np.where(ID[n])[0]
        fold = np.zeros(len(T), bool)
        fold[ref_all[::2]] = True                      # 偶数下标作标定集
        for cal, ev in [(fold, ~fold), (~fold, fold)]:
            r = np.sort(L[n][cal]) if cal.sum() >= 20 else None
            if r is None:
                P[n][ev] = 0.0
                continue
            P[n][ev] = np.searchsorted(r, L[n][ev]) / len(r)

    log(f"标定参考：每节点平均 {ID.sum(1).mean():.0f} 张本地未见样本")
    log(f"本地判别头在本地类上精度 {CORR.sum() / ID.sum():.4f}")

    # ---- 查询构成 ----
    rng = np.random.RandomState(0)
    W_uniform = np.ones_like(L)                        # 每节点 10 类等概率
    W_skew = np.zeros_like(L)
    for n in range(Nn):
        idn = np.where(ID[n])[0]
        odn = rng.choice(np.where(~ID[n])[0], size=len(idn), replace=False)
        W_skew[n, idn] = 1.0
        W_skew[n, odn] = 1.0                           # 本地 50%，外来 50%

    grids = np.round(np.arange(0.0, 0.96, 0.01), 2)
    out = {"setting": dict(nodes=Nn, per_node=len(cls_of[0]),
                           n_test=int(len(T)), n_train=int(len(H))),
           "local_acc": float(CORR.sum() / ID.sum()), "sweeps": {}}

    for wname, W in [("uniform", W_uniform), ("skew50", W_skew)]:
        for sname, SHOULD in [("strict", ~ID), ("err", (~ID) | (ID & ~CORR))]:
            w = W.astype(np.float64)
            tot = w.sum()
            n_should = (w * SHOULD).sum()
            n_local_right = (w * CORR).sum()
            rows = []
            for t in grids:
                defer = (P < t)
                tp = (w * defer * SHOULD).sum()
                fp = (w * defer * ~SHOULD).sum()
                fn = (w * ~defer * SHOULD).sum()
                p, r, f = prf(tp, fp, fn)
                cost = fp * 1.0 + fn * 10.0          # 误拒=1，漏拒=10
                rows.append(dict(tau=float(t), defer=float((w * defer).sum() / tot),
                                 prec=p, rec=r, f1=f, frac_local_rejected=float(t),
                                 cost=float(cost / tot)))
            best_f1 = max(rows, key=lambda x: x["f1"])
            best_cost = min(rows, key=lambda x: x["cost"])
            at5 = min(rows, key=lambda x: abs(x["tau"] - 0.05))
            key = f"{wname}|{sname}"
            out["sweeps"][key] = dict(rows=rows, best_f1=best_f1,
                                      best_cost=best_cost, at5=at5,
                                      should_rate=float(n_should / tot),
                                      local_right=float(n_local_right / tot))
            log("\n" + "=" * 74)
            log(f"=== 查询构成 {wname} / 该拒定义 {sname} ===")
            log(f"  该拒样本占 {n_should / tot:.3f}；本地本可答对占 {n_local_right / tot:.3f}")
            log(f"  {'τ':>5}{'转出率':>9}{'查准':>8}{'查全':>8}{'F1':>8}{'代价':>9}")
            for row in rows[::5]:
                log(f"  {row['tau']:>5.2f}{row['defer']:>9.3f}{row['prec']:>8.3f}"
                    f"{row['rec']:>8.3f}{row['f1']:>8.3f}{row['cost']:>9.4f}")
            log(f"  F1 最优  τ={best_f1['tau']:.2f}  P={best_f1['prec']:.3f} "
                f"R={best_f1['rec']:.3f} F1={best_f1['f1']:.3f} 转出 {best_f1['defer']:.3f}")
            log(f"  代价最优 τ={best_cost['tau']:.2f}  P={best_cost['prec']:.3f} "
                f"R={best_cost['rec']:.3f} 转出 {best_cost['defer']:.3f} 代价 {best_cost['cost']:.4f}")
            log(f"  参照 5% τ=0.05   P={at5['prec']:.3f} R={at5['rec']:.3f} "
                f"F1={at5['f1']:.3f}")

    with open(os.path.join(ROOT, "results", "results_selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("\n已写 results_selfcheck.json")


if __name__ == "__main__":
    main()
