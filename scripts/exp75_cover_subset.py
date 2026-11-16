"""exp75: 让类在节点间有多个副本时，覆盖全部类的小子集训练 body 是否够用

作者观点（本实验要检验的）：真实网络里同类节点很多，同一类由多个节点持有，
因此「覆盖全部类」的子集远小于全网；只让这个子集参与训练，是一种去中心化的
局部训练，不必全网参与。

与 exp72 的差别（关键）：exp72 沿用正文划分（20 节点 × 5 类、100 类无重叠），
任何子集都覆盖不全，故「覆盖全类的子集」在那种划分下等于全网。本实验改用
**有副本的划分**：20 节点 × 20 类、每类恰有 4 个持有者，每类 300 张训练图切成
4 份、每份 75 张给一个持有者。于是 5～6 个节点即可覆盖全部 100 类。

对比：
  none    不训练（基线）
  all20   全部 20 节点参训（数据 100 类 × 300 张）
  cover   贪心集合覆盖选出的最小覆盖集（数据 100 类 × 75 张）
  cover+  覆盖集 + 1 个节点（看边际）
  rand    随机同样多节点（可能覆盖不全），3 个种子

读数：A1_all / A1_seen / A1_unseen / Fisher / 覆盖率
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp32_end2end_all_lda import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100  # noqa: E402
from exp72_subset_producer import (  # noqa: E402
    fisher_ratio, route_a1, train_body, apply_body,
)

torch.manual_seed(0)
np.random.seed(0)

K = 100
NN = 20
KPN = 20            # 每节点 20 类
COPIES = 4          # 每类 4 个持有者
SHARD = 300 // COPIES   # 每份 75 张
N_TEST = 3000
EPOCHS = 300


def log(*a):
    print(*a, flush=True)


def build_partition(seed=0):
    """每类 4 个持有者，每节点恰 20 类，每类 300 张切 4 份。

    返回 cls_of_node[n] = [(c, shard_id), ...]
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)          # 打乱类编号，避免与语义顺序相关
    cls_of_node = {n: [] for n in range(NN)}
    for c in range(K):
        base = int(perm[c]) % NN
        for r in range(COPIES):
            n = (base + r) % NN
            cls_of_node[n].append((c, r))
    for n in cls_of_node:
        cls_of_node[n].sort()
    return cls_of_node


def node_data(H, y, cls_of_node, n):
    """返回节点 n 持有的样本索引（按 (类, 分片) 取）。"""
    idx = []
    for c, r in cls_of_node[n]:
        rows = np.flatnonzero(y == c)
        idx.append(rows[r * SHARD:(r + 1) * SHARD])
    return np.concatenate(idx)


def greedy_cover(cls_of_node, k_max=NN):
    """集合覆盖贪心：每轮选覆盖未覆盖类最多的节点。"""
    uncovered = set(range(K))
    sel, trace = [], []
    while uncovered and len(sel) < k_max:
        best, gain = None, -1
        for n in range(NN):
            if n in sel:
                continue
            g = len({c for c, _ in cls_of_node[n]} & uncovered)
            if g > gain:
                best, gain = n, g
        sel.append(best)
        uncovered -= {c for c, _ in cls_of_node[best]}
        trace.append((best, gain, len(uncovered)))
    return sel, trace


def evaluate(H_new, y, T_new, yT, seen_cls):
    mu, W, d = fit_lda(H_new, y, min(K - 1, H_new.shape[1]))
    Z = (H_new - mu) @ W
    Zt = (T_new - mu) @ W
    protos, pcls = [], []
    for c in range(K):
        Xc = Z[y == c]
        if len(Xc) == 0:
            continue
        protos.append(Xc.mean(0))
        pcls.append(c)
    pred = route_a1(Zt, yT, np.stack(protos), np.array(pcls))
    ok = pred == yT
    seen = np.isin(yT, list(seen_cls))
    return dict(d=int(d), a1_all=float(ok.mean()),
                a1_seen=float(ok[seen].mean()),
                a1_unseen=float(ok[~seen].mean()) if (~seen).any() else None,
                fisher=float(fisher_ratio(Z, y)))


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    keep = np.isin(y, np.arange(K))
    H, y = H[keep], y[keep]
    log(f"训练特征 {H.shape}，测试 {T.shape}")

    cls_of_node = build_partition(0)
    cnt = {c: 0 for c in range(K)}
    for n in range(NN):
        for c, _ in cls_of_node[n]:
            cnt[c] += 1
    assert set(cnt.values()) == {COPIES}, cnt
    per_node = [len({c for c, _ in cls_of_node[n]}) for n in range(NN)]
    log(f"划分：每节点类数 {min(per_node)}～{max(per_node)}，每类副本 {set(cnt.values())}")

    base = evaluate(H, y, T, yT, np.arange(K))
    log(f"  none    A1_all {base['a1_all']:.4f}  Fisher {base['fisher']:.2f}")

    cover, trace = greedy_cover(cls_of_node)
    log("集合覆盖贪心：" + " → ".join(f"n{n}(+{g},剩{u})" for n, g, u in trace))
    log(f"  最小覆盖集 {len(cover)} 个节点：{cover}")

    plans = {"all": list(range(NN)), "cover": cover}
    if len(cover) + 1 <= NN:
        plans["cover+1"] = greedy_cover(cls_of_node)[0] + [
            n for n in range(NN) if n not in cover][:1]
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        plans[f"rand{s+1}"] = sorted(rng.choice(NN, len(cover), replace=False).tolist())

    res = {"baseline": base, "cover_set": cover,
           "cover_trace": [[int(a), int(b), int(c)] for a, b, c in trace],
           "plans": {}}

    for tag, nodes in plans.items():
        t0 = time.time()
        rows = np.concatenate([node_data(H, y, cls_of_node, n) for n in nodes])
        ys_raw = y[rows]
        seen_cls = sorted({c for n in nodes for c, _ in cls_of_node[n]})
        uniq = {c: i for i, c in enumerate(seen_cls)}
        ys = np.array([uniq[c] for c in ys_raw], dtype=np.int64)
        body = train_body(H[rows], ys)
        r = evaluate(apply_body(body, H), y, apply_body(body, T), yT, seen_cls)
        r.update(nodes=nodes, n_nodes=len(nodes), n_classes=len(seen_cls),
                 n_samples=int(len(rows)), sec=round(time.time() - t0, 1))
        res["plans"][tag] = r
        log(f"  {tag:8s} 节点{len(nodes):2d} 类{len(seen_cls):3d} 样本{r['n_samples']:6d}  "
            f"A1_all {r['a1_all']:.4f}  seen {r['a1_seen']:.4f}  "
            f"unseen {('%.4f' % r['a1_unseen']) if r['a1_unseen'] is not None else '   -  '}  "
            f"Fisher {r['fisher']:.2f}  ({r['sec']}s)")

    for tag, r in res["plans"].items():
        res["plans"][tag]["gain_vs_none"] = round((r["a1_all"] - base["a1_all"]) * 100, 2)
    out = os.path.join(ROOT, "results", "results_cover_subset.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"写入 {out}")


if __name__ == "__main__":
    main()
