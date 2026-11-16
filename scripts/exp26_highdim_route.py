"""exp26: 高维会毁掉路由吗？—— 贪婪下降 vs 全局扫描，按维度扫描

之前所有"召回@k"都是**全局排序**的结果（把 100 个原型全比一遍）。
但真正的 P2P 路由是**局部贪婪下降**：只沿 K 个邻居走，可能卡在局部最小。
高维下邻居图质量会变差，两者可能分道扬镳。

测 d = 16 / 32 / 64 / 128 / 384：
  ① greedy_exact  贪婪终点 == 全局最近 vnode
  ② greedy_hit    贪婪终点与全局最近 vnode 同类（路由真正要的）
  ③ 平均跳数 / 卡住率（贪婪终点 ≠ 全局最近）
  ④ 图质量：平均度、跨类边比例、图直径
  ⑤ 拒识阈值的可用性：kNN 距离的变异系数（越接近 0 越可用）
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
    V = V[:, ::-1].astype(np.float32)   # 降序

    cls_ids = np.arange(100)
    protos = np.stack([H[y == c].mean(0) for c in range(100)]).astype(np.float32)

    rng = np.random.RandomState(0)
    K = 8
    res = []

    log(f"  {'d':>4}{'greedy_exact':>14}{'greedy_hit':>12}{'平均跳数':>10}"
        f"{'卡住率':>9}{'平均度':>8}{'跨类边':>9}{'图直径':>8}{'kNN变异':>10}")
    log("-" * 92)

    for dim in [16, 32, 64, 128, 384]:
        W = V[:, :dim]
        Zv = (protos - mu) @ W
        Zq = (T - mu) @ W

        # 全局最近
        D = dist(Zq, Zv)
        nearest = np.argmin(D, 1)

        # KNN 图（对称化）
        Dv = dist(Zv, Zv)
        np.fill_diagonal(Dv, np.inf)
        nbr = np.argsort(Dv, 1)[:, :K]
        adj = [set(nbr[i]) for i in range(100)]
        for i in range(100):
            for j in nbr[i]:
                adj[j].add(i)

        # 图直径（BFS 近似，取平均最远距离）
        diam = 0
        for s in range(0, 100, 5):
            dist_b = {s: 0}
            q = [s]
            while q:
                u = q.pop(0)
                for v_ in adj[u]:
                    if v_ not in dist_b:
                        dist_b[v_] = dist_b[u] + 1
                        q.append(v_)
            if dist_b:
                diam = max(diam, max(dist_b.values()))

        deg = np.mean([len(a) for a in adj])
        # 跨类边
        cross = np.mean([1.0 if cls_ids[j] != cls_ids[i] else 0.0
                         for i in range(100) for j in adj[i]])

        # 贪婪下降：随机起点
        hops_l, exact, hit, stuck = [], 0, 0, 0
        for i in range(len(Zq)):
            start = rng.randint(100)
            cur, h = start, 0
            while h < 30:
                cand = list(adj[cur])
                if not cand:
                    break
                dd = dist(Zq[i:i + 1], Zv[cand])[0]
                b = cand[int(np.argmin(dd))]
                if dd.min() < dist(Zq[i:i + 1], Zv[cur:cur + 1])[0]:
                    cur = b
                    h += 1
                else:
                    break
            hops_l.append(h)
            if cur == nearest[i]:
                exact += 1
            else:
                stuck += 1
            if cls_ids[cur] == yT[i]:
                hit += 1

        n = len(Zq)
        # kNN 距离变异系数（拒识阈值可用性）
        knn_d = np.min(D, 1)
        cv = float(knn_d.std() / knn_d.mean())

        log(f"  {dim:>4}{exact/n:>14.4f}{hit/n:>12.4f}{np.mean(hops_l):>10.2f}"
            f"{stuck/n:>9.4f}{deg:>8.1f}{cross:>9.3f}{diam:>8d}{cv:>10.3f}")
        res.append(dict(d=dim, greedy_exact=exact / n, greedy_hit=hit / n,
                        hops=float(np.mean(hops_l)), stuck=stuck / n,
                        deg=float(deg), cross_class=float(cross), diameter=int(diam),
                        knn_cv=cv))

    with open(os.path.join(ROOT, "results", "results_highdim.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    log("\n  判读：greedy_hit 稳住 = 高维不毁路由；跨类边高 = 邻居表被异类占满；")
    log("        kNN变异接近 0.6+ = 绝对距离阈值不可用（只能比排序，不能定半径）")


if __name__ == "__main__":
    main()
