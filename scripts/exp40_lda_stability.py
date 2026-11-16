"""exp40: 判别投影 W 的稳定性（LDA 口径）—— 换一批拟合数据，W 与坐标会不会变

动机：正文第 4.2.1 节称 W 可一次发布、长期复用。exp10 只在 PCA 口径下验证过，
     而 2026-10-21 起主方法已改为判别投影（LDA），故此处按同一设计重做：
      ① 四套互不相同的样本各自拟合 W，比最近邻同类率
      ② 比两套 W 所张子空间的主角（cos）
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp37_metric_learn import fit_lda  # noqa: E402


def log(*a):
    print(*a, flush=True)


def nn1(mu, W, Hq, Hg, yq, yg):
    Zq = (Hq - mu) @ W
    Zg = (Hg - mu) @ W
    d2 = (Zq ** 2).sum(1)[:, None] + (Zg ** 2).sum(1)[None, :] - 2 * Zq @ Zg.T
    o = d2.argmin(1)
    return float(np.mean(yg[o] == yq)), float(np.mean(yq == yg[o])), o


def main():
    from exp12_proto_route import build_split
    t0 = time.time()
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    yh = ytr[tr_sel].astype(np.int64)
    yt = yte[te_sel].astype(np.int64)
    log(f"train {H.shape} / test {T.shape} / 类数 {len(np.unique(yh))}")

    rng = np.random.RandomState(0)
    n_query, n_gallery = 1000, 5000
    idx = rng.permutation(len(H))
    qi, gi = idx[:n_query], idx[n_query:n_query + n_gallery]
    Hq, Hg = H[qi], H[gi]
    yq, yg = yh[qi], yh[gi]

    d = int(len(np.unique(yh)) - 1)          # 判别方向数 = K-1
    perm = np.random.RandomState(7).permutation(len(H))
    sets = {
        "train_all": (H, yh),
        "test_only": (T, yt),
        "train_halfA": (H[perm[:len(H) // 2]], yh[perm[:len(H) // 2]]),
        "train_halfB": (H[perm[len(H) // 2:]], yh[perm[len(H) // 2:]]),
    }

    fits, rows = {}, []
    for name, (X, yX) in sets.items():
        mu, W, _ = fit_lda(X, yX, d)
        fits[name] = (mu, W)
        acc = nn1(mu, W, Hq, Hg, yq, yg)[0]
        rows.append(dict(fit=name, n=len(X), d=int(W.shape[1]), nn1=acc))
        log(f"用 {name:12s} ({len(X):5d} 张) 拟合 -> d={W.shape[1]} nn1={acc:.4f}")

    log("\n--- 子空间一致性（主角 cos，1.000 表示完全一致）---")
    names = list(fits)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = np.linalg.svd(fits[names[i]][1].T @ fits[names[j]][1], compute_uv=False)
            pairs.append(dict(a=names[i], b=names[j], cos_mean=float(s.mean()),
                              cos_min=float(s.min()),
                              cos_top3=[float(x) for x in s[:3]]))
            log(f"{names[i]:12s} vs {names[j]:12s}  mean_cos={s.mean():.4f}  "
                f"min={s.min():.4f}  前3={np.round(s[:3], 4)}")

    with open(os.path.join(ROOT, "results", "results_lda_stability.json"), "w", encoding="utf-8") as f:
        json.dump(dict(k=d, method="LDA", fit_results=rows, pairs=pairs), f,
                  ensure_ascii=False, indent=2)
    log(f"\n完成 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
