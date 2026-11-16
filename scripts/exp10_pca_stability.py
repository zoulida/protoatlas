"""exp10: PCA 的 W 稳定吗？—— 换一批拟合数据，坐标系会不会变

动机：PCA 的合法性缺口是"W 在目标数据上拟合"。若 W 对拟合样本高度敏感，
     换成公开参照集拟合就会掉点，方案有硬伤；若 W 稳定，则 W 视为数据分布的
     稳定属性，用公开集拟合代价很小，缺口可关。

做法：用 4 种互不相同的样本集拟合 W（16 维），
      ① 比较两两子空间的主角（cos 值越接近 1 越一致）
      ② 在固定的 query/gallery 上比 1-NN 同类率
"""
import os
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def load_labels_aligned():
    import sys
    sys.path.insert(0, ROOT)
    import exp1_route as E

    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    node_classes, node_idx = E.make_shard2(ytr, C["n_nodes"], C["n_classes"],
                                           C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    sel = []
    for n, idx in enumerate(node_idx):
        for k in node_classes[n]:
            sub = idx[ytr[idx] == k]
            take = rng.choice(sub, size=min(C["per_class_train"], len(sub)), replace=False)
            sel.append(take)
    return ytr[np.concatenate(sel)].astype(np.int32)


def fit_pca(X, k):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(cov.astype(np.float64))
    V = V[:, ::-1][:, :k].astype(np.float32)
    return mu, V


def principal_angles(V1, V2):
    s = np.linalg.svd(V1.T @ V2, compute_uv=False)
    return s  # = cos(主角)


def nn1(mu, V, Hq, Hg, yq, yg):
    Zq = (Hq - mu) @ V
    Zg = (Hg - mu) @ V
    d2 = (Zq ** 2).sum(1)[:, None] + (Zg ** 2).sum(1)[None, :] - 2 * Zq @ Zg.T
    o = d2.argmin(1)
    return float(np.mean(yg[o] == yq))


def main():
    t0 = time.time()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y = load_labels_aligned()
    log(f"train {H.shape}  test {T.shape}")

    rng = np.random.RandomState(0)
    n_query, n_gallery = 1000, 5000
    idx = rng.permutation(len(H))
    qi, gi = idx[:n_query], idx[n_query:n_query + n_gallery]
    Hq, Hg = H[qi], H[gi]
    yq, yg = y[qi], y[gi]

    k = 16
    perm = np.random.RandomState(7).permutation(len(H))
    sets = {
        "train_all": H,
        "test_only": T,
        "train_halfA": H[perm[:len(H) // 2]],
        "train_halfB": H[perm[len(H) // 2:]],
    }

    fits = {}
    rows = []
    for name, X in sets.items():
        mu, V = fit_pca(X, k)
        fits[name] = (mu, V)
        acc = nn1(mu, V, Hq, Hg, yq, yg)
        rows.append(dict(fit=name, n=len(X), nn1=acc))
        log(f"用 {name:12s} ({len(X):5d} 张) 拟合 -> nn1={acc:.4f}")

    log("\n--- 子空间一致性（主角 cos，1.000 表示完全一致）---")
    names = list(fits)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = principal_angles(fits[names[i]][1], fits[names[j]][1])
            pairs.append(dict(a=names[i], b=names[j],
                              cos_mean=float(c.mean()), cos_min=float(c.min()),
                              cos_top3=[float(x) for x in c[:3]]))
            log(f"{names[i]:12s} vs {names[j]:12s}  mean_cos={c.mean():.4f}  "
                f"min={c.min():.4f}  前3={np.round(c[:3],4)}")

    with open(os.path.join(ROOT, "results", "results_pca_stability.json"), "w", encoding="utf-8") as f:
        json.dump(dict(k=k, fit_results=rows, pairs=pairs), f, ensure_ascii=False, indent=2)
    log(f"\n完成 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
