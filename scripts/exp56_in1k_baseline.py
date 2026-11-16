"""exp56: ImageNet-1k 千类档的集中式基线（表 5-4 前四行的可溯源版本）

背景：表 5-4 前四行（1-NN 0.7446、最近类均值 0.7758 / 0.7404、线性分类器
0.7356 / 0.9008）在本地结果文件中查不到出处，本脚本按 exp52 的同一划分、
同一特征重算，并加一组与本文 $A^\\star_1$ 同口径的对照。

设定（与 exp52 raw 口径对齐）：
  K = 1000，D = 384，不投影、仅减全体均值
  每类 50 张：前 n_pool=30 张作原型池，其余作查询池；每个种子抽 3000 条查询
  5 个种子（seed 0..4），报均值与标准差

五项读数：
  ① 1-NN，参照为原型池全部 30 张 × 1000 类
  ② 最近类均值，每类 30 张（单次估计）
  ③ 最近类均值，每类有放回抽 15 张，5 次重复
  ④ 线性分类器（闭式 ridge，384 维输入、1000 类输出）
  ⑤ 最近类均值，每类 35 个重采样副本（每副本 15 张有放回）——与本文 A* 同口径

召回@3 两种口径：same（最近 3 个参照本身的类）、dedup（最近 3 个不同类）
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

K = 1000
DIM = 384
SEEDS = (0, 1, 2, 3, 4)
NQ = 3000
REPEAT15 = 5
SCAN = 64
BLK = 400
OUT = os.path.join(ROOT, "results", "results_in1k_baseline_5seed.json")


def log(*a):
    print(*a, flush=True)


def sqdist(A, B):
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    D = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    return D


def ridge_probe(H, y, n_cls, lam=1.0):
    X = H.astype(np.float64)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.zeros((len(X), n_cls), np.float64)
    Y[np.arange(len(X)), y] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def recall3_from_order(srt, cls, yq, n, scan=200):
    """按距离顺序取最近 3 个参照的类（same）与最近 3 个不同类（dedup）"""
    same = np.zeros(n, bool)
    dedup = np.zeros(n, bool)
    for i in range(n):
        row = srt[i]
        same[i] = int(yq[i]) in set(cls[row[:3]].tolist())
        seen = []
        for j in row[:scan]:
            c = int(cls[j])
            if c not in seen:
                seen.append(c)
                if len(seen) == 3:
                    break
        dedup[i] = int(yq[i]) in seen
    return float(same.mean()), float(dedup.mean())


def run_seed(seed, H, y):
    rng = np.random.RandomState(seed)
    idx_by_c = [np.where(y == c)[0] for c in range(K)]
    n_min = min(len(v) for v in idx_by_c)
    n_pool = max(20, int(n_min * 0.6))
    proto_idx, query_idx = [], []
    for c in range(K):
        v = idx_by_c[c]
        rng.shuffle(v)
        proto_idx.append(v[:n_pool])
        query_idx.append(v[n_pool:])
    proto_idx = np.concatenate(proto_idx)
    query_idx = np.concatenate(query_idx)
    mu = H[proto_idx].mean(0).astype(np.float32)
    Zp = (H[proto_idx] - mu)
    yp = y[proto_idx]

    # ⑤ 放在抽查询之前构造，使随机数消耗顺序与 exp52 一致
    #    （exp52：划分 → 造 vnode 副本 → 建图 → 抽查询）
    rep, rep_cls = [], []
    for c in range(K):
        base = Zp[yp == c]
        for row in rng.choice(n_pool, (35, 15), replace=True):
            rep.append(base[row].mean(0))
            rep_cls.append(c)
    rep = np.stack(rep)
    rep_cls = np.array(rep_cls)

    qi = rng.choice(len(query_idx), NQ, replace=False)
    Q = (H[query_idx] - mu)[qi]
    yq = y[query_idx][qi]
    n = len(Q)
    r = {"n_pool": n_pool, "nq": n}

    # ① 1-NN
    t1 = np.empty(n, np.int32)
    s_all = np.empty((n, min(SCAN, Zp.shape[0] - 1)), np.int32)
    for i in range(0, n, BLK):
        d = sqdist(Q[i:i + BLK], Zp)
        k = min(SCAN, Zp.shape[0] - 1)
        part = np.argpartition(d, k, axis=1)[:, :k]
        part = np.take_along_axis(part, np.argsort(np.take_along_axis(d, part, 1), 1), 1)
        s_all[i:i + BLK] = part
        t1[i:i + BLK] = part[:, 0]
    rs, rd = recall3_from_order(s_all, yp, yq, n)
    r["nn1"] = {"top1": float((yp[t1] == yq).mean()), "recall3_same": rs, "recall3_dedup": rd}

    # ② 类均值，每类 n_pool 张
    Cm = np.stack([Zp[yp == c].mean(0) for c in range(K)])
    srt = np.argsort(sqdist(Q, Cm), 1)
    r["mean30"] = {"top1": float((srt[:, 0] == yq).mean()),
                   "recall3": float(np.mean([yq[i] in srt[i, :3] for i in range(n)]))}

    # ③ 类均值，每类有放回 15 张，5 次重复
    a1, a3 = [], []
    for _ in range(REPEAT15):
        Cm = np.stack([Zp[yp == c][rng.choice(n_pool, 15, replace=True)].mean(0)
                       for c in range(K)])
        srt = np.argsort(sqdist(Q, Cm), 1)
        a1.append(float((srt[:, 0] == yq).mean()))
        a3.append(float(np.mean([yq[i] in srt[i, :3] for i in range(n)])))
    r["mean15"] = {"top1": float(np.mean(a1)), "top1_std": float(np.std(a1, ddof=1)),
                   "recall3": float(np.mean(a3))}

    # ④ 线性分类器
    Wr = ridge_probe(Zp, yp, K)
    Sq = np.hstack([Q.astype(np.float64), np.ones((n, 1))]) @ Wr
    srt = np.argsort(-Sq, 1)
    r["linear"] = {"top1": float((srt[:, 0] == yq).mean()),
                   "recall3": float(np.mean([yq[i] in srt[i, :3] for i in range(n)]))}

    # ⑤ 35 个重采样副本（同 A* 口径，副本已在抽查询前构造）
    rep1 = np.empty(n, np.int32)
    same = np.zeros(n, bool)
    dedup = np.zeros(n, bool)
    for i in range(0, n, BLK):
        d = sqdist(Q[i:i + BLK], rep)
        srt = np.argsort(d, 1)
        rep1[i:i + BLK] = srt[:, 0]
        for rr in range(d.shape[0]):
            same[i + rr] = int(yq[i + rr]) in set(rep_cls[srt[rr, :3]].tolist())
            seen = []
            for j in srt[rr, :200]:
                c = int(rep_cls[j])
                if c not in seen:
                    seen.append(c)
                    if len(seen) == 3:
                        break
            dedup[i + rr] = int(yq[i + rr]) in seen
    r["mean35rep"] = {"top1": float((rep_cls[rep1] == yq).mean()),
                      "recall3_same": float(same.mean()),
                      "recall3_dedup": float(dedup.mean())}
    return r


def main():
    H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int32)
    log(f"  特征 {H.shape}，K={int(y.max()) + 1}")

    per = {}
    for seed in SEEDS:
        r = run_seed(seed, H, y)
        per[f"s{seed}"] = r
        log(f"  seed {seed}: 1NN {r['nn1']['top1']:.4f}  mean30 {r['mean30']['top1']:.4f}  "
            f"mean15 {r['mean15']['top1']:.4f}  lin {r['linear']['top1']:.4f}  "
            f"rep35 {r['mean35rep']['top1']:.4f}")

    keys = ["nn1", "mean30", "mean15", "linear", "mean35rep"]
    summ = {}
    for k in keys:
        fields = per["s0"][k].keys()
        summ[k] = {}
        for f in fields:
            vals = [per[f"s{s}"][k][f] for s in SEEDS]
            summ[k][f] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals, ddof=1)), "per_seed": vals}

    log("  === 五种子汇总 ===")
    for k in keys:
        s = summ[k]
        if k == "nn1":
            log(f"  {k:9s} top1 {s['top1']['mean']:.4f}±{s['top1']['std']:.4f}  "
                f"r3same {s['recall3_same']['mean']:.4f}  r3dedup {s['recall3_dedup']['mean']:.4f}")
        elif k == "mean35rep":
            log(f"  {k:9s} top1 {s['top1']['mean']:.4f}±{s['top1']['std']:.4f}  "
                f"r3same {s['recall3_same']['mean']:.4f}  r3dedup {s['recall3_dedup']['mean']:.4f}")
        else:
            log(f"  {k:9s} top1 {s['top1']['mean']:.4f}±{s['top1']['std']:.4f}  "
                f"recall3 {s['recall3']['mean']:.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"K": K, "d": DIM, "seeds": list(SEEDS), "summary": summ,
                   "per_seed": per}, f, ensure_ascii=False, indent=1)
    log(f"  写入 {OUT}")


if __name__ == "__main__":
    main()
