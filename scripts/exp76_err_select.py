"""exp76: 由「答错」驱动选点 —— 错误率能否代替「知道哪些类没被覆盖」

作者设想：若某节点总是区分不了新样本（无论本地判还是路由到别处判），就把它
并入 body 的训练集。本实验检验这个判据的效力与代价。

设定：CIFAR-100，20 节点 × 20 类，每类 4 副本（同 exp75）。
  1. 先只用 4 个节点（覆盖 80 类）训一份 body，制造「有 20 类没被训练看到」的状态；
  2. 在该状态下统计每个节点的端到端错误率（该节点持类的测试样本走路由后的错误率）；
  3. 追加 2 个节点，用四种策略：
       err      错误率最高的 2 个（作者的判据，只需本地可观测的「我答错了」）
       oracle   覆盖未覆盖类最多的 2 个（需要全局标签视图，A2 下不可得，作上界）
       kcenter  坐标空间 k-center 最远的 2 个（无标签，4.3 可行）
       rand     随机 2 个，3 个种子
  4. 用 6 个节点重训，读 A1_all。

若 err 接近 oracle 且明显高于 rand，说明「答错」是一个在 A2 下可表达、
且能替代「类覆盖」的选点信号。
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from exp32_end2end_all_lda import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100  # noqa: E402
from exp72_subset_producer import (  # noqa: E402
    fisher_ratio, route_a1, train_body, apply_body,
)
from exp75_cover_subset import (  # noqa: E402
    K, NN, build_partition, node_data, greedy_cover,
)

torch.manual_seed(0)
np.random.seed(0)

N_TEST = 3000
N_INIT = 4      # 初始参训节点数（覆盖约 80 类）
N_ADD = 2       # 追加节点数


def log(*a):
    print(*a, flush=True)


def fit_and_route(H_new, y, T_new, yT):
    mu, W, d = fit_lda(H_new, y, min(K - 1, H_new.shape[1]))
    Z = (H_new - mu) @ W
    Zt = (T_new - mu) @ W
    protos, pcls = [], []
    for c in range(K):
        Xc = Z[y == c]
        if len(Xc):
            protos.append(Xc.mean(0))
            pcls.append(c)
    P = np.stack(protos)
    pcls = np.array(pcls)
    pred = route_a1(Zt, yT, P, pcls)
    return dict(mu=mu, W=W, Z=Z, Zt=Zt, P=P, pcls=pcls, pred=pred,
                a1=float((pred == yT).mean()), fisher=float(fisher_ratio(Z, y)))


def node_error(r, yT, cls_of_node, n):
    """节点 n 的端到端错误率：其持类的测试样本被判错的比例。"""
    m = np.isin(yT, list({c for c, _ in cls_of_node[n]}))
    if not m.any():
        return 0.0, 0
    return float((r["pred"][m] != yT[m]).mean()), int(m.sum())


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    keep = np.isin(y, np.arange(K))
    H, y = H[keep], y[keep]

    cls_of_node = build_partition(0)
    cover, _ = greedy_cover(cls_of_node)
    init = cover[:N_INIT]                      # 贪心顺序的前 4 个，覆盖 80 类
    covered0 = sorted({c for n in init for c, _ in cls_of_node[n]})
    log(f"初始 {N_INIT} 节点 {init}，覆盖 {len(covered0)} 类，缺 {K-len(covered0)} 类")

    # 基线：不训练
    base = fit_and_route(H, y, T, yT)
    log(f"  none      A1 {base['a1']:.4f}")

    # 第一阶段：只用 init 训练
    rows0 = np.concatenate([node_data(H, y, cls_of_node, n) for n in init])
    uniq0 = {c: i for i, c in enumerate(covered0)}
    body0 = train_body(H[rows0], np.array([uniq0[c] for c in y[rows0]], np.int64))
    r0 = fit_and_route(apply_body(body0, H), y, apply_body(body0, T), yT)
    log(f"  init{N_INIT}     A1 {r0['a1']:.4f}")

    # 每个节点的错误率（在当前状态下观测）
    errs = np.array([node_error(r0, yT, cls_of_node, n)[0] for n in range(NN)])
    counts = np.array([node_error(r0, yT, cls_of_node, n)[1] for n in range(NN)])
    order = np.argsort(-errs)
    log("  节点错误率（降序）：" + " ".join(
        f"n{n}:{errs[n]:.3f}/{counts[n]}" for n in order[:8]))

    # oracle：覆盖未覆盖类最多的节点
    miss = set(range(K)) - set(covered0)
    gain = {n: len({c for c, _ in cls_of_node[n]} & miss) for n in range(NN)}
    oracle = sorted(gain, key=lambda n: -gain[n])[:N_ADD]
    log(f"  oracle 选 {oracle}（各覆盖未覆盖类 {[gain[n] for n in oracle]} 个）")

    # k-center（坐标空间，用初始状态的坐标）
    cen = np.stack([r0["Z"][np.isin(y, list({c for c, _ in cls_of_node[n]}))].mean(0)
                    for n in range(NN)])
    sel = list(init)
    dist = np.min([((cen - cen[s]) ** 2).sum(1) for s in sel], axis=0)
    while len(sel) < N_INIT + N_ADD:
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, ((cen - cen[j]) ** 2).sum(1))
    kcen = [n for n in sel if n not in init][:N_ADD]
    log(f"  kcenter 追加 {kcen}")

    err_add = [int(n) for n in order if n not in init][:N_ADD]
    log(f"  err 追加 {err_add}（错误率 {[round(errs[n],3) for n in err_add]}）")

    plans = {"err": err_add, "oracle": oracle, "kcenter": kcen}
    for s in range(3):
        rng = np.random.default_rng(200 + s)
        plans[f"rand{s+1}"] = sorted(rng.choice(
            [n for n in range(NN) if n not in init], N_ADD, replace=False).tolist())

    res = dict(baseline=base["a1"], init=r0["a1"], init_nodes=init,
               covered0=len(covered0), node_err=errs.tolist(),
               node_cnt=counts.tolist(), oracle=oracle, err=err_add,
               kcenter=kcen, plans={})

    for tag, add in plans.items():
        nodes = init + add
        t0 = time.time()
        rows = np.concatenate([node_data(H, y, cls_of_node, n) for n in nodes])
        seen = sorted({c for n in nodes for c, _ in cls_of_node[n]})
        uniq = {c: i for i, c in enumerate(seen)}
        body = train_body(H[rows], np.array([uniq[c] for c in y[rows]], np.int64))
        r = fit_and_route(apply_body(body, H), y, apply_body(body, T), yT)
        res["plans"][tag] = dict(add=add, n_nodes=len(nodes), n_classes=len(seen),
                                 n_samples=int(len(rows)), a1=r["a1"],
                                 fisher=r["fisher"], sec=round(time.time() - t0, 1))
        log(f"  {tag:8s} +{add} 类{len(seen):3d} 样本{len(rows):6d}  "
            f"A1 {r['a1']:.4f}  ({res['plans'][tag]['sec']}s)")

    out = os.path.join(ROOT, "results", "results_err_select.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"写入 {out}")


if __name__ == "__main__":
    main()
