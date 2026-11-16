"""exp5: 距离集中现象 vs 坐标维度
问题：RayNet 的贪婪路由靠"比距离"，维度越高所有点对距离越接近，
      排序失去意义。本脚本量化这个退化，给出可用的维度上限。
指标：
  contrast = d_nn / d_median   -> 越接近 1 越离散不开
  nn1_acc  = 最近邻同类比例
  top3_rec = 前 3 邻居的同类覆盖率
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
    """feat_train.npy 的行序 = exp1 里 tr_sel 的顺序，必须按同样流程重建标签。"""
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
    sel = np.concatenate(sel)
    return ytr[sel].astype(np.int32)


def main():
    t0 = time.time()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    y = load_labels_aligned()
    assert len(y) == len(H), f"标签 {len(y)} 与特征 {len(H)} 不匹配"

    rng = np.random.RandomState(0)
    n_gallery = 5000
    n_query = 1000
    idx = rng.permutation(len(H))
    qi, gi = idx[:n_query], idx[n_query:n_query + n_gallery]
    Hq, HQ = H[qi], H[gi]
    yq, yg = y[qi], y[gi]

    mu = H.mean(0, keepdims=True)
    Hc = H - mu
    cov = (Hc.T @ Hc) / (len(Hc) - 1)
    w, V = np.linalg.eigh(cov.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)
    ev = w[::-1]
    log(f"PC1 解释方差 {ev[0]/ev.sum()*100:.1f}%  "
        f"前2 {ev[:2].sum()/ev.sum()*100:.1f}%  "
        f"前8 {ev[:8].sum()/ev.sum()*100:.1f}%  "
        f"前16 {ev[:16].sum()/ev.sum()*100:.1f}%")

    out = []
    for dim in [2, 4, 8, 16, 32, 64, 128, 384]:
        Vk = V[:, :dim]
        Zq = (Hq - mu) @ Vk
        Zg = (HQ - mu) @ Vk
        # 平方距离
        d2 = (Zq ** 2).sum(1)[:, None] + (Zg ** 2).sum(1)[None, :] - 2 * Zq @ Zg.T
        d2 = np.maximum(d2, 0)
        # 按距离升序
        order = np.argsort(d2, axis=1)[:, :5]
        dmin = np.sqrt(d2[np.arange(n_query), order[:, 0]])
        dmed = np.sqrt(np.median(d2, axis=1))
        contrast = float(np.mean(dmin / dmed))
        nn1 = float(np.mean(yg[order[:, 0]] == yq))
        top3 = float(np.mean([np.mean(yg[order[i, :3]] == yq[i]) for i in range(n_query)]))
        # 最近邻距离的分布离散度：不同 query 之间 d_nn 的变异系数
        cv = float(np.std(dmin) / np.mean(dmin))
        out.append(dict(dim=dim, contrast=contrast, nn1_acc=nn1,
                        top3_rec=top3, nn_dist_cv=cv,
                        var_explained=float(ev[:dim].sum() / ev.sum())))
        log(f"d={dim:4d}  方差={out[-1]['var_explained']*100:5.1f}%  "
            f"contrast={contrast:.3f}  nn1={nn1:.3f}  top3={top3:.3f}  cv={cv:.3f}")

    with open(os.path.join(ROOT, "results", "results_conc.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"完成 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
