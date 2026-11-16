"""exp30: 候选有了，怎么变成单一答案？—— 裁决器对比

exp29 结果：
  候选召回@3 = 0.9918，候选中有人答对 = 0.9900（理论上界）
  但"距离加权投票"只有 0.9262，比硬 top-1 的 0.9442 还差。

说明信息是够的，是裁决器不对。这里测 6 种裁决规则：
  1 nearest      取最近候选节点的预测（= 硬 top-1）
  2 max_margin   取所有候选节点里 margin 最高的
  3 max_prob     取 softmax 概率最高的
  4 w_dist       exp(-d/T) 加权投票
  5 w_prob*prior 概率 × 距离先验
  6 reject_first 最近候选若 margin 低于阈值，改用第 2 候选
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def train_linear(X, y, n_cls, epochs=300, lr=0.05, wd=1e-4):
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
    return m


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]))

    # 所有节点对所有测试样本的 logits
    L = np.zeros((Nn, len(T), 2), np.float32)
    with torch.no_grad():
        for n in range(Nn):
            for i in range(0, len(T), 1024):
                L[n, i:i + 1024] = heads[n](torch.from_numpy(T[i:i + 1024])).numpy()

    def to_global(n, lg_row):
        """节点局部 logits → 全局 10 类 logits（未见的类填 -inf）"""
        g = np.full(10, -1e9, np.float32)
        g[cls_of_node[n]] = lg_row
        return g

    # 原型与候选
    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zp = (P - mu[0]) @ W
    Zq = (T - mu[0]) @ W
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2 * Zq @ Zp.T, 0))
    order = np.argsort(D, 1)

    K = 3
    cand = []
    for i in range(len(T)):
        seen, out = set(), []
        for j in order[i]:
            c = int(pcls[j])
            if c not in seen:
                seen.add(c)
                out.append((c, int(pnode[j]), float(D[i, j])))
            if len(out) == K:
                break
        cand.append(out)

    in_cand = np.mean([y_te[i] in [c for c, _, _ in cand[i]] for i in range(len(T))])
    oracle = np.mean([
        any(y_te[i] in cls_of_node[n]
            and cls_of_node[n][int(np.argmax(L[n, i]))] == y_te[i]
            for c, n, _ in cand[i])
        for i in range(len(T))])
    log(f"候选召回@{K} {in_cand:.4f}   候选内有人答对（oracle）{oracle:.4f}")

    res = {}

    def report(name, pred):
        a = float(np.mean(pred == y_te))
        log(f"  {name:<26}{a:.4f}")
        res[name] = a
        return a

    log(f"\n  {'裁决规则':<26}{'准确率':>8}")
    log("  " + "-" * 36)

    # 1 nearest
    p1 = np.array([int(np.argmax(to_global(cand[i][0][1], L[cand[i][0][1], i])))
                   for i in range(len(T))])
    report("1 nearest（硬 top-1）", p1)

    # 2 max_margin
    p2 = []
    for i in range(len(T)):
        best, bm = None, -1e9
        for c, n, d in cand[i]:
            lg = L[n, i]
            mg = lg.max() - np.partition(lg, -2)[-2]
            if mg > bm:
                bm, best = mg, c
        p2.append(best)
    report("2 max_margin", np.array(p2))

    # 3 max_prob
    p3 = []
    for i in range(len(T)):
        best, bp = None, -1
        for c, n, d in cand[i]:
            lg = L[n, i]
            e = np.exp(lg - lg.max())
            pr = (e / e.sum()).max()
            if pr > bp:
                bp, best = pr, c
        p3.append(best)
    report("3 max_prob（softmax）", np.array(p3))

    # 4 w_dist
    p4 = []
    for i in range(len(T)):
        ds = np.array([c[2] for c in cand[i]])
        wt = np.exp(-ds / (ds.mean() + 1e-8))
        sc = np.zeros(10)
        for (c, n, d), ww in zip(cand[i], wt):
            sc[int(np.argmax(to_global(n, L[n, i])))] += ww
        p4.append(int(np.argmax(sc)))
    report("4 距离加权投票", np.array(p4))

    # 5 prob × 距离先验
    p5 = []
    for i in range(len(T)):
        ds = np.array([c[2] for c in cand[i]])
        prior = np.exp(-ds / (ds.mean() + 1e-8))
        sc = np.zeros(10)
        for (c, n, d), pw in zip(cand[i], prior):
            g = to_global(n, L[n, i])
            e = np.exp(g - g.max())
            sc += pw * (e / e.sum())
        p5.append(int(np.argmax(sc)))
    report("5 概率×距离先验", np.array(p5))

    # 6 reject_first：最近候选 margin 低则换第 2 个
    margins = np.array([[L[n, i].max() - np.partition(L[n, i], -2)[-2]
                         for c, n, d in cand[i]] for i in range(len(T))])
    thr = np.percentile(margins[:, 0], 20)
    p6 = np.array([cand[i][0][0] if margins[i, 0] >= thr else cand[i][1][0]
                   for i in range(len(T))])
    report("6 低 margin 换次选", p6)

    log(f"\n  参考：集中式 0.9630 / FedAvg 0.9672（exp29）")
    log(f"  裁决上界（oracle）{oracle:.4f}")

    with open(os.path.join(ROOT, "results", "results_arbiter.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
