# -*- coding: utf-8 -*-
"""exp60: 近邻表建表通信量（RayNet 式交换填充）

协议（对应正文 4.3.1「近邻表在加入之后自组织」）：
  每个 vnode 初始只认识 1 个随机邻居；此后反复随机挑一个已知邻居，
  与之交换整张表（各 κ 条记录），合并后保留距自身最近的 κ 条，直至表内容不再变化。
  一次交换的字节数 = 2 × κ × (4d + ℓ)，取 ℓ=8。

输出：每节点平均交换次数、每节点平均字节、全局总字节，以及收敛轮数。
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
ELL = 8


def load_protos(d):
    """取 ImageNet-1k 的 1000 类原型作为 vnode 坐标（判别投影前用原始特征，维度按 d 截断）。"""
    F = np.load(os.path.join(CACHE, "feat_in1k.npy"), mmap_mode="r")
    F = np.asarray(F, dtype=np.float32)
    # feat_in1k 是全部样本，按 1000 类等分（每类 50 张，见第 5 章设定）
    n = len(F) // 1000 * 1000
    F = F[:n].reshape(1000, -1, F.shape[1]).mean(1)
    if d < F.shape[1]:
        # 取前 d 维并重新归一化，模拟降维后的坐标分布
        F = F[:, :d]
    return np.ascontiguousarray(F.astype(np.float64))


def build(M, d, kappa, seed=0):
    rng = np.random.RandomState(seed)
    X = load_protos(d)
    if M <= 1000:
        idx = rng.choice(len(X), size=M, replace=False)
        Z = X[idx]
    else:
        rep = int(np.ceil(M / len(X)))
        Z = np.repeat(X, rep, axis=0)[:M]
        Z = Z + rng.normal(0, 0.01, size=Z.shape)   # 同族副本的微小抖动
    return Z


def dmat(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def simulate(M, d, kappa, max_round=60, seed=0):
    Z = build(M, d, kappa, seed)
    rng = np.random.RandomState(seed + 7)
    tabs = [np.array([rng.randint(M)], dtype=np.int64) for _ in range(M)]
    # 去掉自环
    for i in range(M):
        while tabs[i][0] == i:
            tabs[i][0] = rng.randint(M)
    Zl = Z
    n_exch = 0
    rounds = 0
    for r in range(max_round):
        changed = 0
        order = rng.permutation(M)
        for i in order:
            ti = tabs[i]
            j = int(ti[rng.randint(len(ti))])
            tj = tabs[j]
            cand = np.unique(np.concatenate([ti, tj, [j]]))
            dd = np.linalg.norm(Zl[cand] - Zl[i], axis=1)
            keep = cand[np.argsort(dd)[:kappa]]
            n_exch += 1
            if len(keep) != len(ti) or not np.array_equal(np.sort(keep), np.sort(ti)):
                changed += 1
            tabs[i] = keep
        rounds = r + 1
        if changed == 0:
            break
    msg = 4 * d + ELL
    per_node_b = 2 * kappa * msg * (n_exch / M)
    return dict(M=M, d=d, kappa=kappa, rounds=rounds,
                exch_total=n_exch, exch_per_node=n_exch / M,
                msg_b=msg, per_node_kb=per_node_b / 1024,
                total_mb=n_exch * 2 * kappa * msg / 1024 / 1024,
                avg_deg=float(np.mean([len(t) for t in tabs])))


if __name__ == "__main__":
    out = []
    for M in (100, 300, 1000, 3000):
        r = simulate(M, 384, 16, seed=1)
        out.append(r)
        print(f"M={r['M']:>6} d=384 κ=16  轮数={r['rounds']:>3}  "
              f"交换/节点={r['exch_per_node']:>6.1f}  "
              f"每节点={r['per_node_kb']:>8.1f} KB  合计={r['total_mb']:>8.1f} MB  "
              f"平均度={r['avg_deg']:.1f}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "results_table_comm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("写入 results_table_comm.json")
