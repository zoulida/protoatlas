"""exp79: 去掉判别投影 W，只用 φ 的原特征做最近原型路由，精度会掉多少

作者设想：W 只为压缩维度而存在，压缩既不提精度也不影响精度，故可整件删除，
坐标直接取 φ(x)。

本脚本直接对撞这个设想。同一份 ImageNet-1k 特征、同一划分、同一查询集，
只改「坐标怎么算」：

  甲  不投影，坐标 = φ(x)，384 维原样做欧氏最近原型
  乙  投影，坐标 = W^T φ(x)，维数 d = min(K-1, 384)（正文现在的做法）

读两个量，都按类去重（同一类只算一次）：
  A1        最近原型的命中率
  召回@3    前 3 个不同类里是否命中真类

若甲的读数与乙接近，删 W 只需承担维数带来的通信代价；
若差距随 K 拉大，说明 W 的作用不止压缩。

  K = 10, 50, 100, 200, 500, 1000，每档 3 个种子
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp32_end2end_all_lda import fit_lda  # noqa: E402

N_PROTO = 30
N_QUERY = 20
KS = [10, 50, 100, 200, 500, 1000]
SEEDS = 3


def log(*a):
    print(*a, flush=True)


def evaluate(Zp, yp, Zq, yq, pcls, k=3):
    """Zp/Zq 已含原型与查询，pcls 为原型对应的类，按类去重后取前 k 个"""
    P = np.stack([Zp[yp == c].mean(0) for c in pcls]).astype(np.float64)
    Q = Zq.astype(np.float64)
    Dm = np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :]
                            - 2 * Q @ P.T, 0))
    order = np.argsort(Dm, 1)
    pcls = np.asarray(pcls)
    a1 = 0
    hit = 0
    for i in range(len(Q)):
        seen = []
        for j in order[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.append(c)
            if len(seen) == 1 and c == yq[i]:
                a1 += 1
            if len(seen) >= k:
                break
        if yq[i] in seen:
            hit += 1
    return a1 / len(Q), hit / len(Q)


def run(H, y, K, seed):
    rng = np.random.default_rng(seed)
    cls = rng.choice(1000, K, replace=False)
    idx = []
    for c in cls:
        rows = np.where(y == c)[0]
        if len(rows) < N_PROTO + N_QUERY:
            return None
        idx.append(rows[:N_PROTO + N_QUERY])
    idx = np.concatenate(idx)
    Hp = H[idx].astype(np.float32)
    yp = y[idx].copy()
    is_q = np.zeros(len(idx), bool)
    for c in cls:
        rows = np.where(y == c)[0][:N_PROTO + N_QUERY]
        is_q[np.isin(idx, rows[N_PROTO:])] = True
    Hq = Hp[is_q]
    yq = yp[is_q]
    Hp = Hp[~is_q]
    yp = yp[~is_q]

    # 甲：不投影
    a1_raw, r3_raw = evaluate(Hp, yp, Hq, yq, list(cls))

    # 乙：判别投影
    mu, W, d = fit_lda(Hp, yp, min(K - 1, Hp.shape[1]))
    a1_w, r3_w = evaluate((Hp - mu) @ W, yp, (Hq - mu) @ W, yq, list(cls))

    return dict(K=K, d=int(d), a1_raw=a1_raw, a1_w=a1_w,
                r3_raw=r3_raw, r3_w=r3_w)


def main():
    H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "in1k_y.npy"))
    log(f"特征 {H.shape}，类数 {len(np.unique(y))}")
    log(f"{'K':>6} {'d':>4} | {'A1 不投影':>9} {'A1 投影':>9} {'差':>7} "
        f"| {'R@3 不投影':>10} {'R@3 投影':>9} {'差':>7}")

    out = []
    for K in KS:
        rows = [r for r in (run(H, y, K, s) for s in range(SEEDS)) if r]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "K" and k != "d"}
        d = rows[0]["d"]
        out.append(dict(K=K, d=d, **m))
        log(f"{K:6d} {d:4d} | {m['a1_raw']:9.4f} {m['a1_w']:9.4f} "
            f"{m['a1_w']-m['a1_raw']:+7.4f} | {m['r3_raw']:10.4f} {m['r3_w']:9.4f} "
            f"{m['r3_w']-m['r3_raw']:+7.4f}")

    with open(os.path.join(ROOT, "results", "results_noW.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("写入 results_noW.json")


if __name__ == "__main__":
    main()
