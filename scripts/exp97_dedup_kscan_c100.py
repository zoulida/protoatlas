# -*- coding: utf-8 -*-
"""exp97：按类去重的效果——CIFAR-100 设定（与 exp96 的 CIFAR-10 对照）

exp96 取 CIFAR-10：10 类、20 节点各 2 类、M=40、每类 4 副本。
本脚本换成第 5.1 节 100 类档的构造：CIFAR-100、200 节点各 2 类、M=400、
每类 4 副本（每副本 125 张），查询取测试集 10 000 张。

要回答的问题：类数从 10 提到 100 后，按类去重相对不去重的增益是更大还是更小。
两个口径与 exp96 完全一致：
    raw@k    取距离最近的 k 个 vnode，真类在其中即算命中（名额按 vnode 计）
    dedup@k  按类去重后取 k 个不同的类（名额按类计）
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

K = 100
N_NODE = 2 * K
PER_COPY = 125
KS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
OUT = os.path.join(ROOT, "results", "results_dedup_kscan_c100.json")


def d2(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


def main():
    from exp53_c100_end2end_lda import load_c100
    from exp1_route import make_shard2

    H_all, y_all, _, _ = load_c100()
    m = y_all < K
    H_all, y_all = H_all[m], y_all[m]
    T = np.load(os.path.join(CACHE, "feat_c100_test10k.npy")).astype(np.float32)
    y_te = np.load(os.path.join(CACHE, "c100_test10k.npz"))["y"].astype(np.int64)

    nc, ni = make_shard2(y_all, N_NODE, K, 2, seed=0)
    protos, pcls = [], []
    for n in range(N_NODE):
        for c in nc[n]:
            idx = ni[n][y_all[ni[n]] == c]
            if len(idx) == 0:
                continue
            take = idx[:min(PER_COPY, len(idx))]
            protos.append(H_all[take].mean(0))
            pcls.append(c)
    P = np.stack(protos).astype(np.float32)
    pcls = np.array(pcls, np.int64)
    print(f"vnode 数 {len(P)}，类数 {len(set(pcls.tolist()))}，查询 {len(T)}", flush=True)

    D = d2(T, P)
    o = np.argsort(D, 1)
    nq = len(y_te)

    rows = []
    for k in KS:
        top = o[:, :k]
        raw = float((pcls[top] == y_te[:, None]).any(1).mean())
        hit = 0
        for i in range(nq):
            seen = []
            for j in o[i]:
                c = int(pcls[j])
                if c not in seen:
                    seen.append(c)
                if len(seen) == k:
                    break
            hit += int(int(y_te[i]) in seen)
        ded = hit / nq
        rows.append(dict(k=k, raw=raw, dedup=ded))
        print(f"k={k:2d}  不去重 {raw:.4f}   去重 {ded:.4f}", flush=True)

    diff3 = float(np.mean([len(set(pcls[o[i, :3]].tolist())) for i in range(nq)]))
    print(f"未去重时前 3 名平均含 {diff3:.2f} 个不同类")

    out = dict(setting=dict(dataset="CIFAR-100", n_nodes=N_NODE, per_node=2,
                            copies=4, n_vnode=int(len(P)), n_test=int(nq)),
               diff3=diff3, rows=rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入 results_dedup_kscan_c100.json")


if __name__ == "__main__":
    main()
