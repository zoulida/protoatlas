"""临时诊断：贪婪下降为什么会停在不该停的地方（与 exp28 同口径）
d=64、100 类均值原型、k=8 对称 kNN 图。
对"没走到全局最近原型"的查询，打印：
  ① 终点 p 与全局最近 g 的距离差
  ② p 的 8 个邻居到 q 的距离（最近的那个比 p 远多少）
  ③ p 到 g 在图上的 BFS 跳数（>1 说明路是通的但要走先变远的一步）
  ④ 最短路第一步是否比 p 更远
"""
import os
import sys
import numpy as np
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def main():
    import pandas as pd
    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:3000]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    V = V[:, ::-1].astype(np.float32)
    protos = np.stack([H[y == c].mean(0) for c in range(100)]).astype(np.float32)
    W = V[:, :64]
    Zv = (protos - mu) @ W
    Zq = (T - mu) @ W

    Dv = dist(Zv, Zv)
    np.fill_diagonal(Dv, np.inf)
    K = 8
    nbr = np.argsort(Dv, 1)[:, :K]
    adj = [set(nbr[i]) for i in range(100)]
    for i in range(100):
        for j in nbr[i]:
            adj[j].add(i)
    adj = [sorted(a) for a in adj]

    Dg = dist(Zq, Zv)
    rng = np.random.RandomState(0)
    LOCAL = 8
    qn, hop_gap, nb_extra, bfs_hops, first_farther = [], [], [], [], []
    stuck_n = 0
    N = len(Zq)
    for i in range(N):
        q = Zq[i]
        loc = rng.choice(100, LOCAL, replace=False)
        s = int(loc[np.argsort(dist(q[None, :], Zv[loc])[0])[:1]][0])
        cur, h = s, 0
        while h < 30:
            cand = adj[cur]
            dd = dist(q[None, :], Zv[cand])[0]
            if dd.min() < float(dist(q[None, :], Zv[cur:cur + 1])[0,0]):
                cur, h = cand[int(np.argmin(dd))], h + 1
            else:
                break
        g = int(np.argmin(Dg[i]))
        if cur == g:
            continue
        stuck_n += 1
        dp = float((dist(q[None, :], Zv[cur:cur + 1])[0,0]))
        dg = float(Dg[i, g])
        hop_gap.append(dp - dg)
        dnb = dist(q[None, :], Zv[adj[cur]])[0]
        nb_extra.append(float(dnb.min()) - dp)
        # BFS 从 cur 到 g
        prev = {cur: None}
        dq_ = deque([cur])
        found = False
        while dq_ and not found:
            u = dq_.popleft()
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    if v == g:
                        found = True
                        break
                    dq_.append(v)
        if not found:
            bfs_hops.append(-1)
            continue
        # 回溯最短路
        path, x = [], g
        while x is not None:
            path.append(x)
            x = prev[x]
        path.reverse()
        bfs_hops.append(len(path) - 1)
        if len(path) > 2:
            first = path[1]
            first_farther.append((float(dist(q[None, :], Zv[first:first + 1])[0,0])) - dp)

    hop_gap = np.array(hop_gap)
    nb_extra = np.array(nb_extra)
    print(f"查询 {N}，未到全局最近 {stuck_n}（{stuck_n / N:.1%}）")
    print(f"终点比全局最近原型远：中位 {np.median(hop_gap):.2f}，均值 {hop_gap.mean():.2f}，"
          f"p90 {np.percentile(hop_gap, 90):.2f}")
    print(f"终点最近的邻居仍比终点远：中位 {np.median(nb_extra):.2f}，"
          f"p10 {np.percentile(nb_extra, 10):.2f}，最大 {nb_extra.max():.2f}")
    bh = np.array(bfs_hops)
    print(f"BFS 跳数(终点→全局最近)：不可达 {int((bh < 0).sum())}，"
          f"1 跳 {int((bh == 1).sum())}，2 跳 {int((bh == 2).sum())}，"
          f">=3 跳 {int((bh >= 3).sum())}，中位 {np.median(bh[bh >= 0]) if (bh>=0).any() else -1}")
    ff = np.array(first_farther)
    if len(ff):
        print(f"最短路第一步比终点更远的比例：{(ff > 0).mean():.1%}，"
              f"远多少 中位 {np.median(ff):.2f}")


if __name__ == "__main__":
    main()
