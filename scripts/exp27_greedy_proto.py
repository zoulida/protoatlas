"""exp27: 实际协议形态 —— 起点策略 × 到达后是否扩散

exp26 发现：随机起点 + 纯贪婪，greedy_hit 只有 0.38~0.67，比全局排序低 13~18 点。
差距来自两处，必须分开量：
  ① 起点太差（随机落点离目标远，贪婪容易进错盆地）
  ② 没有扩散（到达局部最小后只报一个，同类的其他 vnode 被漏掉）

四种组合：
  A 随机起点 + 纯贪婪        （最悲观）
  B 本地择优起点 + 纯贪婪     （真实：查询方已有 K 个邻居，先本地选最近的）
  C 随机起点 + 一跳扩散
  D 本地择优起点 + 一跳扩散   （推荐协议）
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
    K = 8
    LOCAL = 8          # 查询方本地已知的 vnode 数

    res = {}

    for dim in [16, 64, 128]:
        W = V[:, :dim]
        Zv = (protos - mu) @ W
        Zq = (T - mu) @ W
        Dg = dist(Zq, Zv)
        nearest = np.argmin(Dg, 1)
        Dv = dist(Zv, Zv)
        np.fill_diagonal(Dv, np.inf)
        nbr = np.argsort(Dv, 1)[:, :K]
        adj = [set(nbr[i]) for i in range(100)]
        for i in range(100):
            for j in nbr[i]:
                adj[j].add(i)
        adj = [sorted(a) for a in adj]

        stats = {k: dict(hit=[], r3=[], hops=[]) for k in ["A", "B", "C", "D"]}

        for i in range(len(Zq)):
            q = Zq[i]

            def greedy(start):
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

            def expand(cur, k=3):
                pool = [cur] + adj[cur]
                dd = dist(q[None, :], Zv[pool])[0]
                o = np.argsort(dd)
                seen, picked = [], []
                for j in o:
                    c = int(cls_ids[pool[j]])
                    if c not in seen:
                        seen.append(c)
                        picked.append(c)
                    if len(picked) == k:
                        break
                return picked

            # 两种起点
            s_rand = int(rng.randint(100))
            loc = rng.choice(100, LOCAL, replace=False)
            s_best = int(loc[int(np.argmin(dist(q[None, :], Zv[loc])[0]))])

            for tag, s, do_exp in [("A", s_rand, False), ("B", s_best, False),
                                   ("C", s_rand, True), ("D", s_best, True)]:
                cur, h = greedy(s)
                stats[tag]["hops"].append(h)
                if do_exp:
                    cand3 = expand(cur)
                    stats[tag]["r3"].append(int(yT[i]) in cand3)
                    stats[tag]["hit"].append(cand3[0] == yT[i])
                else:
                    stats[tag]["hit"].append(int(cls_ids[cur]) == yT[i])
                    stats[tag]["r3"].append(int(cls_ids[cur]) == yT[i])

        log("\n" + "=" * 72)
        log(f"=== d = {dim} ===  (K={K} 邻居, 本地已知 {LOCAL} 个 vnode)")
        log(f"  {'策略':<26}{'hit@1':>9}{'召回@3':>9}{'平均跳数':>10}")
        glob1 = float(np.mean(cls_ids[nearest] == yT))
        log(f"  {'全局扫描（上界）':<26}{glob1:>9.4f}{'—':>9}{'—':>10}")
        names = {"A": "A 随机起点+纯贪婪", "B": "B 本地择优+纯贪婪",
                 "C": "C 随机起点+一跳扩散", "D": "D 本地择优+一跳扩散"}
        row = {"global_top1": glob1}
        for tag in ["A", "B", "C", "D"]:
            s = stats[tag]
            h1, r3 = float(np.mean(s["hit"])), float(np.mean(s["r3"]))
            log(f"  {names[tag]:<26}{h1:>9.4f}{r3:>9.4f}{np.mean(s['hops']):>10.2f}")
            row[tag] = dict(hit=h1, r3=r3, hops=float(np.mean(s["hops"])))
        res[dim] = row

    with open(os.path.join(ROOT, "results", "results_greedy_proto.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    log("\n  判读：B−A = 起点质量的价值；C−A = 一跳扩散的价值；D 与全局上界的差 = 分布式损失")


if __name__ == "__main__":
    main()
