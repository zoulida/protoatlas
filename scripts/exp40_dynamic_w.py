"""exp40: 类别持续增加时，W 该不该重算？怎么才能不打断在线路由？

场景（用户提出）：系统长期运行 → 节点越来越多 → 新类不断出现 → 路由变差。
需要一个"动态更新 LDA 坐标、但不影响在线路由"的协议。

本实验回答四个问题：
  A. 完全不更新 W（沿用旧版），新类加进来后掉多少？
  B. 重算 W（全量重发布）能挽回多少？
  C. 能不能只广播一个 d×d 正交矩阵 + d 维平移（而不是重发所有原型坐标）？
     这条决定了"更新协议"的带宽量级。
  D. 有没有**本地可算**的报警信号？（不能等全局准确率，那需要全局标签）

设定：CIFAR-100 特征，按固定乱序逐批加入 20 类：20 → 40 → 60 → 80 → 100。
每个阶段用"截至目前出现过的类"训练 LDA，在"已出现类的测试样本"上评估路由。
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


def route_acc(Zq, Zp, pcls, yte):
    D2 = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
    o = D2.argmin(1)
    pred = pcls[o]
    hit = pred == yte
    return float(hit.mean()), hit, o


def procrustes_rot(Wold, Wnew):
    """求正交 R 使 Wold·R ≈ Wnew（距离在正交变换下不变，故坐标可直接转换）"""
    d = Wold.shape[1]
    A = Wold[:, :d].T @ Wnew[:, :d]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    return R


def subspace_cos(Wa, Wb, k=3):
    """主子空间的主角余弦"""
    d = min(k, Wa.shape[1], Wb.shape[1])
    Qa, _ = np.linalg.qr(Wa[:, :d])
    Qb, _ = np.linalg.qr(Wb[:, :d])
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(s.mean())


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
        log(f"=== 目标维度 d={d_want}，CIFAR-100 逐批加类 ===")
        log(f"  {'阶段':>6}{'类数':>5}{'d':>4}{'旧W路由':>9}{'旧W@新类':>10}"
            f"{'新W路由':>9}{'旋转近似':>9}{'子空间余弦':>10}")
        log("  " + "-" * 72)

        prev = None                      # (mu, W, protos, pcls)
        rows = []
        for t, K in enumerate(stages):
            cls_seen = list(order[:K])
            mask_tr = np.isin(y, cls_seen)
            Hs, ys = H[mask_tr], y[mask_tr]
            d_eff = min(d_want, K - 1)

            mu_t, W_t, _ = fit_lda(Hs, ys, d_eff, whiten=True)
            protos, pcls = [], []
            for c in cls_seen:
                protos.append(H[y == c].mean(0))
                pcls.append(c)
            P = np.stack(protos).astype(np.float32)
            pcls = np.array(pcls)

            mask_te = np.isin(yt, cls_seen)
            Tq, yq = T[mask_te], yt[mask_te]

            Zp_new = (P - mu_t) @ W_t
            Zq_new = (Tq - mu_t) @ W_t
            acc_new, _, _ = route_acc(Zq_new, Zp_new, pcls, yq)

            # ---- 旧 W（不更新）----
            if prev is None:
                acc_old, acc_old_newcls, cos = acc_new, acc_new, 1.0
            else:
                mu_o, W_o, _, cls_old = prev
                Zp_o = (P - mu_o) @ W_o
                Zq_o = (Tq - mu_o) @ W_o
                acc_old, hit_o, _ = route_acc(Zq_o, Zp_o, pcls, yq)
                newm = np.isin(yq, [c for c in cls_seen if c not in cls_old])
                acc_old_newcls = float(hit_o[newm].mean()) if newm.sum() else float("nan")
                cos = subspace_cos(W_o, W_t, k=3)

            # ---- 只发正交矩阵 + 平移：能否替代重发布 ----
            if prev is None:
                acc_rot = acc_new
                n_floats_rot, n_floats_full = 0, 384 * d_eff
            else:
                mu_o, W_o, _, _ = prev
                d_common = min(W_o.shape[1], W_t.shape[1])
                R = procrustes_rot(W_o[:, :d_common], W_t[:, :d_common])
                # 旧坐标 → 新坐标：z_new ≈ z_old·R + δ,  δ=(μ_old−μ_new)·W_new
                delta = (mu_o - mu_t) @ W_t[:, :d_common]
                Zp_rot = (P - mu_o) @ W_o[:, :d_common] @ R + delta
                Zq_rot = (Tq - mu_o) @ W_o[:, :d_common] @ R + delta
                acc_rot, _, _ = route_acc(Zq_rot, Zp_rot, pcls, yq)
                n_floats_rot = d_common * d_common + d_common
                n_floats_full = 384 * W_t.shape[1]

            log(f"  {t:>6}{K:>5}{d_eff:>4}{acc_old:>9.4f}{acc_old_newcls:>10.4f}"
                f"{acc_new:>9.4f}{acc_rot:>9.4f}{cos:>10.4f}")
            rows.append(dict(stage=t, K=K, d=d_eff, stale=acc_old,
                             stale_newcls=acc_old_newcls, fresh=acc_new,
                             rotation=acc_rot, cos=cos,
                             kb_rot=n_floats_rot * 4 / 1024,
                             kb_full=n_floats_full * 4 / 1024))
            prev = (mu_t, W_t, P, cls_seen)

        res[f"d{d_want}"] = rows

    # ---------- D. 本地可算的报警信号 ----------
    log("")
    log("=" * 92)
    log("=== D. 本地报警信号：每个节点用自己的样本探针（路由回自己算不算命中）===")
    log("    节点 = 单类持有者。探针命中率与全局准确率是否同步下降？")
    log(f"  {'类数':>5}{'d':>4}{'全局路由(旧W)':>14}{'全库真值':>10}"
        f"{'探针命中(旧W)':>14}{'探针命中(新W)':>14}")
    log("  " + "-" * 64)

    probe_rows = []
    prev = None
    for K in stages:
        cls_seen = list(order[:K])
        mask_tr = np.isin(y, cls_seen)
        Hs, ys = H[mask_tr], y[mask_tr]
        d_eff = min(16, K - 1)
        mu_t, W_t, _ = fit_lda(Hs, ys, d_eff, whiten=True)
        protos = np.stack([H[y == c].mean(0) for c in cls_seen]).astype(np.float32)

        mask_te = np.isin(yt, cls_seen)
        Tq, yq = T[mask_te], yt[mask_te]

        def probe(mu_u, W_u):
            Zp = (protos - mu_u) @ W_u
            Zq = (Tq - mu_u) @ W_u
            D2 = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
            pred = np.array(cls_seen)[D2.argmin(1)]
            # 探针：只看"送回的类是不是自己的类"，逐节点平均
            rates = [float(np.mean(pred[yq == c] == c)) for c in cls_seen
                     if (yq == c).sum() > 0]
            return float(np.mean(rates)), float(np.mean(pred == yq))

        if prev is None:
            p_old, g_old = probe(mu_t, W_t)
            g_true = g_old
        else:
            mu_o, W_o = prev
            p_old, g_old = probe(mu_o, W_o)
            g_true = probe(mu_t, W_t)[1]
        p_new, _ = probe(mu_t, W_t)
        log(f"  {K:>5}{d_eff:>4}{g_old:>14.4f}{g_true:>10.4f}"
            f"{p_old:>14.4f}{p_new:>14.4f}")
        probe_rows.append(dict(K=K, d=d_eff, global_stale=g_old,
                               global_fresh=g_true, probe_stale=p_old,
                               probe_fresh=p_new))
        prev = (mu_t, W_t)

    res["probe"] = probe_rows
    with open(os.path.join(ROOT, "results", "results_dynamic_w.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n已保存 results_dynamic_w.json")


if __name__ == "__main__":
    main()
