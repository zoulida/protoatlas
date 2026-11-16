"""exp42: 加类时能不能"只追加新方向、旧坐标补零"，从而免掉全量重投影？

背景：exp41 证明「坐标对齐（R+δ）方案无效」——位移与原型最近间距同量级，
且 25% 的类未迁移就掉 7.5 点。于是只剩两条路：
  1. 整版切换（所有人重投影，贵在协调）
  2. **追加式升级**：旧基不动，新方向接在后面，旧坐标补零 —— 不重投影

本实验测第 2 条能不能用：
  W_aug = [ W_old | W_extra ]，W_extra 是新问题里与 W_old 正交的判别方向
  旧 vnode 坐标 = [z_old | 0]，新 vnode 坐标 = [(h-μ)·W_old | (h-μ)·W_extra]
  可选：旧维乘一个标量 α 补偿白化尺度漂移
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


def route(Zq, Zp, pcls, yq):
    D2 = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
    return float(np.mean(pcls[D2.argmin(1)] == yq))


def orth_against(X, B):
    """把 X 的列投影到 B 的正交补，再正交化"""
    Y = X - B @ (B.T @ X)
    Q, _ = np.linalg.qr(Y)
    return Q[:, :X.shape[1]]


def main():
    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int64)
    yt = np.load(os.path.join(CACHE, "c100.npz"))["y"].astype(np.int64)

    rng = np.random.RandomState(7)
    order = rng.permutation(100)
    stages = [20, 40, 60, 80, 100]

    res = {}
    for d0 in [16]:
        for extra in [4, 8]:
            log("=" * 96)
            log(f"=== 旧基 d={d0}，每次升级追加 {extra} 个正交新方向 ===")
            log(f"  {'类数':>5}{'维度':>6}{'整版重算':>10}{'不更新':>9}"
                f"{'追加(α=1)':>11}{'追加(α*)':>10}{'补回报废率':>11}")
            log("  " + "-" * 64)

            prev = None
            rows = []
            for K in stages:
                cls_seen = list(order[:K])
                mtr = np.isin(y, cls_seen)
                mu_t, W_t, _ = fit_lda(H[mtr], y[mtr], min(d0 + 4 * extra, K - 1), whiten=True)
                P = np.stack([H[y == c].mean(0) for c in cls_seen]).astype(np.float32)
                pcls = np.array(cls_seen)
                mte = np.isin(yt, cls_seen)
                Tq, yq = T[mte], yt[mte]

                # fresh：整版重算（用它自己的最优 d0 维）
                mu_f, W_f, _ = fit_lda(H[mtr], y[mtr], d0, whiten=True)
                fresh = route((Tq - mu_f) @ W_f, (P - mu_f) @ W_f, pcls, yq)

                if prev is None:
                    prev = (mu_f, W_f, cls_seen)
                    log(f"  {K:>5}{d0:>6}{fresh:>10.4f}{'—':>9}{'—':>11}{'—':>10}{'—':>11}")
                    rows.append(dict(K=K, d=d0, fresh=fresh))
                    continue

                mu_o, W_o, cls_old = prev
                n_old_cls = len(cls_old)
                is_old = np.isin(pcls, cls_old)

                # 不更新
                stale = route((Tq - mu_o) @ W_o, (P - mu_o) @ W_o, pcls, yq)

                # 追加：新问题方向里，与 W_o 正交的前 extra 个
                W_ex = orth_against(W_t[:, :extra], W_o)
                W_aug = np.hstack([W_o, W_ex])
                mu_u = mu_o

                Zp_full = (P - mu_u) @ W_aug
                Zq_full = (Tq - mu_u) @ W_aug
                Zp_old = np.hstack([(P - mu_u) @ W_o, np.zeros((len(P), W_ex.shape[1]), np.float32)])
                # 查询侧永远是"已迁移"的节点 → 用完整坐标
                # 旧 vnode 用补零坐标（等价于它从未重报）
                def mix(alpha):
                    Zp = Zp_full.copy()
                    Zp[is_old] = Zp_old[is_old]
                    Zp[is_old, :W_o.shape[1]] *= alpha
                    Zq = Zq_full.copy()
                    Zq[:, :W_o.shape[1]] *= alpha
                    return route(Zq, Zp, pcls, yq)

                acc1 = mix(1.0)
                best = (None, -1)
                for a in np.linspace(0.6, 1.6, 21):
                    v = mix(float(a))
                    if v > best[1]:
                        best = (float(a), v)

                rec = 1.0 - (fresh - acc1) / max(1e-9, fresh - stale)
                log(f"  {K:>5}{W_aug.shape[1]:>6}{fresh:>10.4f}{stale:>9.4f}"
                    f"{acc1:>11.4f}{best[1]:>10.4f}{rec*100:>10.1f}%")
                rows.append(dict(K=K, d=int(W_aug.shape[1]), fresh=fresh, stale=stale,
                                 append=acc1, append_best=best[1], alpha=best[0]))
                prev = (mu_f, W_f, cls_seen)
            res[f"d{d0}_extra{extra}"] = rows

    with open(os.path.join(ROOT, "results", "results_dynamic_w3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    log("\n已保存 results_dynamic_w3.json")


if __name__ == "__main__":
    main()
