"""exp41: W 版本切换的迁移代价 —— 只广播 R(d×d) + δ(d) 时，新旧坐标能否混用？

exp40 的 C 组写错了：正交变换下距离不变，故"转换后"与"不转换"路由结果必然相同。
真正要测的是：
  1. 位移误差：用 (R, δ) 把旧坐标搬到新基，与真新坐标差多少（相对原型间典型距离）
  2. 混合迁移：f 比例的类还停在旧基（已转换），其余已是新基 —— 准确率怎么衰减
  3. 符号稳定性：LDA 特征向量符号会翻转，重算两次是否给出可比的基
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)
from exp37_metric_learn import fit_lda


def log(*a, **kw):
    print(*a, flush=True, **kw)


def procrustes(Wold, Wnew):
    A = Wold.T @ Wnew
    U, S, Vt = np.linalg.svd(A)
    return U @ Vt


def route(Zq, Zp, pcls, yq):
    D2 = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
    return float(np.mean(pcls[D2.argmin(1)] == yq))


def main():
    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int64)
    yt = np.load(os.path.join(CACHE, "c100.npz"))["y"].astype(np.int64)

    rng = np.random.RandomState(7)
    order = rng.permutation(100)
    stages = [20, 40, 60, 80, 100]
    res = {}

    for d_want in [16, 32]:
        log("=" * 92)
        log(f"=== d={d_want}：版本切换的位移与混合迁移 ===")
        log(f"  {'类数':>5}{'d':>4}{'位移/范数':>11}{'位移/原型间距':>15}"
            f"{'f=0':>8}{'f=.25':>8}{'f=.5':>8}{'f=.75':>8}{'f=1':>8}")
        log("  " + "-" * 78)

        prev = None
        rows = []
        for K in stages:
            cls_seen = list(order[:K])
            mtr = np.isin(y, cls_seen)
            d_eff = min(d_want, K - 1)
            mu_t, W_t, _ = fit_lda(H[mtr], y[mtr], d_eff, whiten=True)
            P = np.stack([H[y == c].mean(0) for c in cls_seen]).astype(np.float32)
            pcls = np.array(cls_seen)
            mte = np.isin(yt, cls_seen)
            Tq, yq = T[mte], yt[mte]

            Zp_new = (P - mu_t) @ W_t
            Zq_new = (Tq - mu_t) @ W_t

            if prev is None:
                log(f"  {K:>5}{d_eff:>4}{'—':>11}{'—':>15}{'—':>8}{'—':>8}{'—':>8}{'—':>8}"
                    f"{route(Zq_new, Zp_new, pcls, yq):>8.4f}")
                rows.append(dict(K=K, d=d_eff, f1=route(Zq_new, Zp_new, pcls, yq)))
                prev = (mu_t, W_t)
                continue

            mu_o, W_o = prev
            dc = min(W_o.shape[1], W_t.shape[1])
            if W_o.shape[1] != W_t.shape[1]:
                # 维度变化 = 硬断裂，旧坐标无法映射进新基（这条本身是结论）
                acc0 = route(Zq_new, Zp_new, pcls, yq)
                log(f"  {K:>5}{d_eff:>4}{'硬断裂':>11}{f'{W_o.shape[1]}->{W_t.shape[1]}':>15}"
                    f"{acc0:>8.4f}{'—':>8}{'—':>8}{'—':>8}{'—':>8}")
                rows.append(dict(K=K, d=d_eff, hard_break=True,
                                 d_prev=int(W_o.shape[1]), f0=acc0))
                prev = (mu_t, W_t)
                continue
            R = procrustes(W_o[:, :dc], W_t[:, :dc])
            delta = (mu_o - mu_t) @ W_t[:, :dc]

            Zp_conv = (P - mu_o) @ W_o[:, :dc] @ R + delta
            Zq_conv = (Tq - mu_o) @ W_o[:, :dc] @ R + delta

            # 位移误差（相对新基下的坐标范数与原型间最近距离）
            disp = np.linalg.norm(Zp_conv - Zp_new, axis=1)
            scale = np.linalg.norm(Zp_new, axis=1)
            D2p = ((Zp_new[:, None, :] - Zp_new[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(D2p, np.inf)
            nn = np.sqrt(D2p.min(1))
            r1 = float((disp / (scale + 1e-9)).mean())
            r2 = float((disp / (nn + 1e-9)).mean())

            # 混合迁移：f 比例的类仍用旧基（已转换），其余用新基
            accs = {}
            n_cls = len(cls_seen)
            old_mask_cls = np.zeros(n_cls, bool)
            n_old = int(round(0.25 * n_cls))
            old_mask_cls[rng.choice(n_cls, n_old, replace=False)] = True   # 固定 25% 的类"未迁移"
            for f, use_old in [(0.0, np.zeros(n_cls, bool)),
                               (0.25, old_mask_cls),
                               (0.5, old_mask_cls),
                               (0.75, old_mask_cls),
                               (1.0, np.ones(n_cls, bool))]:
                if f in (0.5, 0.75):
                    # 扩大未迁移比例
                    k = int(round(f * n_cls))
                    m2 = np.zeros(n_cls, bool)
                    m2[rng.choice(n_cls, k, replace=False)] = True
                    use_old = m2
                Zp_mix = np.where(use_old[:, None], Zp_conv, Zp_new)
                # 查询侧永远用新基（新加入的节点已迁移）
                accs[f] = route(Zq_new, Zp_mix, pcls, yq)

            log(f"  {K:>5}{d_eff:>4}{r1:>11.4f}{r2:>15.3f}"
                + "".join(f"{accs[f]:>8.4f}" for f in [0.0, 0.25, 0.5, 0.75, 1.0]))
            rows.append(dict(K=K, d=d_eff, disp_norm=r1, disp_nn=r2, **{f"f{k}": v for k, v in accs.items()}))
            prev = (mu_t, W_t)
        res[f"d{d_want}"] = rows

    # ---------- 符号 / 基稳定性 ----------
    log("")
    log("=" * 92)
    log("=== 符号与基稳定性：同一批数据、不同抽样重算两次，W 差多少 ===")
    log(f"  {'K':>5}{'d':>4}{'符号翻转数':>12}{'正交对齐后余弦均值':>20}{'未对齐余弦均值':>17}")
    log("  " + "-" * 60)
    stab = []
    for K in [40, 100]:
        cls_seen = list(order[:K])
        mtr = np.isin(y, cls_seen)
        idx = np.where(mtr)[0]
        A = rng.choice(idx, len(idx) // 2, replace=False)
        B = np.setdiff1d(idx, A)
        d = min(16, K - 1)
        _, Wa, _ = fit_lda(H[A], y[A], d, whiten=True)
        _, Wb, _ = fit_lda(H[B], y[B], d, whiten=True)
        # 未对齐：逐列符号归一后的余弦
        cos_raw = np.abs((Wa * Wb).sum(0) / (np.linalg.norm(Wa, axis=0) * np.linalg.norm(Wb, axis=0) + 1e-12))
        # 符号翻转：以绝对值最大分量为正做规范化（否则 LDA 符号本来就不定）
        def canon(W):
            s = np.sign(W[np.abs(W).argmax(0), np.arange(W.shape[1])])
            return W * s
        Ca, Cb = canon(Wa), canon(Wb)
        flips = int((np.sign((Ca * Cb).sum(0)) < 0).sum())
        Rm = procrustes(Wa, Wb)
        cos_al = np.abs((Wa @ Rm * Wb).sum(0) /
                        (np.linalg.norm(Wa @ Rm, axis=0) * np.linalg.norm(Wb, axis=0) + 1e-12))
        log(f"  {K:>5}{d:>4}{flips:>12}{cos_al.mean():>20.4f}{cos_raw.mean():>17.4f}")
        stab.append(dict(K=K, d=d, sign_flips=flips,
                         cos_aligned=float(cos_al.mean()), cos_raw=float(cos_raw.mean())))
    res["stability"] = stab

    with open(os.path.join(ROOT, "results", "results_dynamic_w2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    log("\n已保存 results_dynamic_w2.json")


if __name__ == "__main__":
    main()
