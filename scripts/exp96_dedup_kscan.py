# -*- coding: utf-8 -*-
"""exp96：按类去重的效果随候选数的变化（图 5-3 重做）

原图把「384 维 / 主成分 16 维」两种特征处理并列，主成分一路已从正文撤掉，
本脚本只保留 384 维原样，改扫候选数 k：

    raw@k    取距离最近的 k 个 vnode，真类在其中即算命中（不去重，名额按 vnode 计）
    dedup@k  按类去重后取 k 个不同的类，真类在其中即算命中

两者在 k=1 处相同，随 k 增大而分开，分开的幅度即去重带来的增益。

设定沿用原实验：CIFAR-10，20 节点各 2 类，每类 4 个副本，共 40 个 vnode。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20)
OUT = os.path.join(ROOT, "results", "results_dedup_kscan.json")


def d2(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


def main():
    from exp13_dedup_reject import build
    H, T, P, pcls, pnode, y_tr, y_te, tr_node, nc, C = build()
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

    # 未去重时前 3 名的不同类数（保留一个构成读数，正文可引）
    diff3 = float(np.mean([len(set(pcls[o[i, :3]].tolist())) for i in range(nq)]))
    print(f"未去重时前 3 名平均含 {diff3:.2f} 个不同类")

    out = dict(setting=dict(dataset="CIFAR-10", n_nodes=20, per_node=2,
                            copies=4, n_vnode=int(len(P)), n_test=int(nq)),
               diff3=diff3, rows=rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入 results_dedup_kscan.json")


if __name__ == "__main__":
    main()
