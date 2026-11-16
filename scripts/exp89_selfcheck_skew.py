"""exp89: 查询构成偏斜到"本地能答 95%~97%"时，自评阈值该取多少

exp88 的构成是"外来 80%"，与真实应用相反：一个节点主要收到自己业务范围内的
查询，需要求助的只有百分之几。本实验把构成改成 3% 与 5%，重扫阈值。

加权方式：本地样本权重 = (1-r)/|ID|，外来样本权重 = r/|外|，r 为求助占比。
指标：
  leak  : 外来样本留在本地作答的比例（每个都是一次错答，直接进端到端）
  misfwd: 本地本可答对却被送出去的比例（只多走一趟，不损失精度）
  E2E   : 本地答对 + 送走后链路答对
  ratio : 每送走 1 个真该求助的，连带送走几个本可本地答对的 —— 比查准更直观
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def main():
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    y_te = yte[te_sel]
    P = np.load(os.path.join(CACHE, "tmp_P88.npy"))
    PRED = np.load(os.path.join(CACHE, "tmp_PRED88.npy"))
    ID = np.load(os.path.join(CACHE, "tmp_ID88.npy"))
    Nn, Nt = P.shape
    CORR = (PRED == y_te[None, :]) & ID

    out = {}
    grids = np.round(np.arange(0.0, 0.96, 0.01), 2)
    for r in (0.03, 0.05):
        w = np.zeros_like(P, dtype=np.float64)
        for n in range(Nn):
            idn = np.where(ID[n])[0]
            odn = np.where(~ID[n])[0]
            w[n, idn] = (1 - r) / len(idn)
            w[n, odn] = r / len(odn)
        tot = w.sum()
        rows = []
        for t in grids:
            defer = (P < t)
            loc = ~defer
            leak = (w * loc * ~ID).sum() / max((w * ~ID).sum(), 1)
            misfwd = (w * defer * ID).sum() / max((w * ID).sum(), 1)
            right_local = (w * loc * CORR).sum() / tot
            fwd = (w * defer).sum() / tot
            e_lo = (right_local + fwd * 0.94)
            e_hi = (right_local + fwd * 0.9552)
            tp = (w * defer * ~ID).sum()
            fp = (w * defer * ID).sum()
            rows.append(dict(tau=float(t), fwd=float(fwd), leak=float(leak),
                             misfwd=float(misfwd), e2e_lo=float(e_lo),
                             e2e_hi=float(e_hi),
                             prec=float(tp / (tp + fp)) if tp + fp > 0 else 0.0,
                             rec=float(tp / max((w * ~ID).sum(), 1)),
                             ratio=float(fp / tp) if tp > 0 else 0.0,
                             wrong=float((w * loc * ~ID).sum() / tot)))
        best = max(rows, key=lambda x: x["e2e_lo"])
        # 端到端进入平台的最小 τ（与本构成下最高 E2E 相差 < 0.2 点）
        peak = best["e2e_lo"]
        plateau = min(r2["tau"] for r2 in rows if r2["e2e_lo"] >= peak - 0.002)
        out[f"r={r}"] = dict(rows=rows, best=best, plateau_tau=float(plateau))

        print("\n" + "=" * 88)
        print(f"=== 求助占比 {r:.0%}（本地能答 {1-r:.0%}）===")
        print(f"  全本地作答 E2E={rows[0]['e2e_lo']:.4f}；全送走 E2E=0.9552"
              f"（本列以 0.94 计送走答对率，故右侧并列两版）")
        print(f"  {'τ':>5}{'转出率':>8}{'外来漏放':>9}{'本地误送':>9}"
              f"{'连带倍数':>9}{'E2E(0.94)':>11}{'E2E(.9552)':>12}")
        for row in rows[::5]:
            print(f"  {row['tau']:>5.2f}{row['fwd']:>8.3f}{row['leak']:>9.3f}"
                  f"{row['misfwd']:>9.3f}{row['ratio']:>9.2f}"
                  f"{row['e2e_lo']:>11.4f}{row['e2e_hi']:>12.4f}")
        print(f"  最优 τ={best['tau']:.2f}  E2E={best['e2e_lo']:.4f}"
              f"  转出 {best['fwd']:.3f}  外来漏放 {best['leak']:.3f}"
              f"  本地误送 {best['misfwd']:.3f}  连带 {best['ratio']:.2f}")
        print(f"  平台起点 τ={plateau:.2f}（离峰值不到 0.2 个点）")

    with open(os.path.join(ROOT, "results", "results_selfcheck_skew.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写 results_selfcheck_skew.json")


if __name__ == "__main__":
    main()
