"""exp77: 「答错」能否被本地可算的几何量替代 —— 半径与余量对节点错误率的解释力

exp76 发现节点错误率在节点间几乎均匀（0.206–0.224），无法用来选点。
本实验问：若把「错没错」换成「散不散」，能否得到一个本地可算、无需标签、
且有区分度的信号？

在 exp76 的 init4 状态下，为每个节点算三个几何量（均在投影空间、只用本地样本）：
  rad    该节点各类的类内半径均值（样本到类原型的距离）
  marg   该类原型到最近异类原型的距离均值
  ratio  marg / rad，即类间余量与类内半径之比
再看它们与该节点端到端错误率的秩相关。
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from exp32_end2end_all_lda import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100  # noqa: E402
from exp72_subset_producer import train_body, apply_body  # noqa: E402
from exp75_cover_subset import K, NN, build_partition, node_data, greedy_cover  # noqa: E402
from exp76_err_select import fit_and_route, node_error  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

N_TEST = 3000


def geom_per_node(Z, Zt, y, yT, cls_of_node):
    """返回每节点的 rad / marg / ratio，只用投影后的坐标。"""
    protos = {c: Z[y == c].mean(0) for c in range(K)}
    rad, marg = {}, {}
    for c in range(K):
        d = np.sqrt(((Z[y == c] - protos[c]) ** 2).sum(1))
        rad[c] = float(d.mean())
        others = np.stack([protos[o] for o in range(K) if o != c])
        marg[c] = float(np.sqrt(((others - protos[c]) ** 2).sum(1)).min())
    out = {}
    for n in range(NN):
        cs = [c for c, _ in cls_of_node[n]]
        out[n] = dict(rad=float(np.mean([rad[c] for c in cs])),
                      marg=float(np.mean([marg[c] for c in cs])),
                      ratio=float(np.mean([marg[c] / max(rad[c], 1e-9) for c in cs])))
    return out


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    keep = np.isin(y, np.arange(K))
    H, y = H[keep], y[keep]

    cls_of_node = build_partition(0)
    cover, _ = greedy_cover(cls_of_node)
    init = cover[:4]
    covered0 = sorted({c for n in init for c, _ in cls_of_node[n]})

    rows0 = np.concatenate([node_data(H, y, cls_of_node, n) for n in init])
    uniq0 = {c: i for i, c in enumerate(covered0)}
    body0 = train_body(H[rows0], np.array([uniq0[c] for c in y[rows0]], np.int64))
    r0 = fit_and_route(apply_body(body0, H), y, apply_body(body0, T), yT)

    errs = np.array([node_error(r0, yT, cls_of_node, n)[0] for n in range(NN)])
    g = geom_per_node(r0["Z"], r0["Zt"], y, yT, cls_of_node)
    rad = np.array([g[n]["rad"] for n in range(NN)])
    marg = np.array([g[n]["marg"] for n in range(NN)])
    ratio = np.array([g[n]["ratio"] for n in range(NN)])
    ncov = np.array([len({c for c, _ in cls_of_node[n]} - set(covered0)) for n in range(NN)])

    print(f"节点错误率  均值 {errs.mean():.4f}  标准差 {errs.std():.4f}  "
          f"极差 {errs.max()-errs.min():.4f}")
    for name, v in (("rad", rad), ("marg", marg), ("ratio", ratio), ("缺类数", ncov)):
        rho, p = spearmanr(v, errs)
        print(f"  {name:6s} 与错误率的 Spearman rho = {rho:+.3f}  (p={p:.3f})  "
              f"自身极差 {v.max()-v.min():.4f}")

    out = os.path.join(ROOT, "results", "results_err_geom.json")
    json.dump(dict(err=errs.tolist(), rad=rad.tolist(), marg=marg.tolist(),
                   ratio=ratio.tolist(), ncov=ncov.tolist()),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
