"""exp28: 分布式损失能否补回？—— 扩散半径 × 起点数

exp27 结论：本地择优 + 一跳扩散，召回@3 比全局扫描低约 10~11 点（与 d 无关）。
本实验在 d=64（100 类推荐配置）上测：
  D  1 起点 + 1 跳扩散      （基线）
  E  1 起点 + 2 跳扩散
  F  3 起点并行 + 1 跳扩散
  G  3 起点并行 + 2 跳扩散
  H  3 起点并行 + 2 跳 + top-5 输出
成本用"实际询问的 vnode 数"衡量（等价于通信量）。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    y = d["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:3000]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)
    protos = np.stack([H[y == c].mean(0) for c in range(100)]).astype(np.float32)
    cls_ids = np.arange(100)
    rng = np.random.RandomState(0)
    K, LOCAL = 8, 8

    dim = 64
    W = V[:, :dim]
    Zv = (protos - mu) @ W
    Zq = (T - mu) @ W
    Dg = dist(Zq, Zv)
    Dv = dist(Zv, Zv)
    np.fill_diagonal(Dv, np.inf)
    nbr = np.argsort(Dv, 1)[:, :K]
    adj = [set(nbr[i]) for i in range(100)]
    for i in range(100):
        for j in nbr[i]:
            adj[j].add(i)
    adj = [sorted(a) for a in adj]

    # 全局扫描的上界（同维度）
    glob_order = np.argsort(Dg, 1)

    def global_recall(k):
        hit = 0
        for i in range(len(yT)):
            seen = []
            for j in glob_order[i]:
                c = int(cls_ids[j])
                if c not in seen:
                    seen.append(c)
                if len(seen) == k:
                    break
            if int(yT[i]) in seen:
                hit += 1
        return hit / len(yT)

    log("=" * 78)
    log(f"d={dim}（100 类，K={K}）。全局扫描上界："
        f"@3 {global_recall(3):.4f}  @5 {global_recall(5):.4f}")
    log(f"  {'方案':<30}{'召回@3':>9}{'召回@5':>9}{'询问节点':>10}{'跳数':>8}")

    def greedy(q, start):
        cur, h = start, 0
        while h < 30:
            cand = adj[cur]
            dd = dist(q[None, :], Zv[cand])[0]
            b = cand[int(np.argmin(dd))]
            if dd.min() < dist(q[None, :], Zv[cur:cur + 1])[0]:
                cur, h = b, h + 1
            else:
                break
        return cur, h

    def pool_of(cur, radius):
        s = {cur}
        frontier = {cur}
        for _ in range(radius):
            nxt = set()
            for u in frontier:
                nxt |= set(adj[u])
            frontier = nxt - s
            s |= nxt
        return sorted(s)

    variants = [
        ("D 1起点 + 1跳 + top3", 1, 1, 3),
        ("E 1起点 + 2跳 + top3", 1, 2, 3),
        ("F 3起点 + 1跳 + top3", 3, 1, 3),
        ("G 3起点 + 2跳 + top3", 3, 2, 3),
        ("H 3起点 + 2跳 + top5", 3, 2, 5),
    ]
    res = {"global_r3": global_recall(3), "global_r5": global_recall(5)}
    for label, ns, radius, kout in variants:
        r3 = r5 = 0
        qn, hp = [], []
        for i in range(len(Zq)):
            q = Zq[i]
            loc = rng.choice(100, LOCAL, replace=False)
            starts = [int(loc[j]) for j in np.argsort(dist(q[None, :], Zv[loc])[0])[:ns]]
            pool, hsum = set(), 0
            for s in starts:
                cur, h = greedy(q, s)
                hsum += h
                pool |= set(pool_of(cur, radius))
            pool = sorted(pool)
            qn.append(len(pool))
            hp.append(hsum / ns)
            dd = dist(q[None, :], Zv[pool])[0]
            o = np.argsort(dd)
            seen = []
            for j in o:
                c = int(cls_ids[pool[j]])
                if c not in seen:
                    seen.append(c)
                if len(seen) >= 5:
                    break
            r3 += int(int(yT[i]) in seen[:3])
            r5 += int(int(yT[i]) in seen[:5])
        n = len(Zq)
        log(f"  {label:<30}{r3/n:>9.4f}{r5/n:>9.4f}{np.mean(qn):>10.1f}{np.mean(hp):>8.2f}")
        res[label] = dict(r3=r3 / n, r5=r5 / n,
                          asked=float(np.mean(qn)), hops=float(np.mean(hp)))

    with open(os.path.join(ROOT, "results", "results_greedy_recover.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    log("\n  判读：与全局上界逐项对比。询问节点数 ≈ 通信量（每个 4~120 字节）")


if __name__ == "__main__":
    main()
