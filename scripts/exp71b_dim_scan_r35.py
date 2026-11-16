"""exp71b: 与表 5-2 同口径（每类 35 个重采样副本）的维数扫描

exp71 用的是每类 1 个原型（30 张均值），绝对读数低于表 5-2 的 A*_1=0.751
（后者每类 35 个副本，最近副本命中的机会更多）。本脚本把副本数补齐，
使 d=384 一档能与表 5-2 对上，再看饱和点是否随副本数后移。
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
K, R, NSUB = 1000, 35, 15
DIMS = [64, 128, 192, 256, 384]
SEEDS = (0, 1)
NQ = 3000
BLK = 200
OUT = os.path.join(ROOT, "results", "results_dim_scan_r35.json")


def log(*a):
    print(*a, flush=True)


H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
y = np.load(os.path.join(CACHE, "in1k_y.npy"))
res = {d: {"a1": [], "a3": []} for d in DIMS}

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    idx = [np.where(y == c)[0] for c in range(K)]
    proto, query = [], []
    for c in range(K):
        v = idx[c].copy()
        rng.shuffle(v)
        proto.append(v[:30])
        query.append(v[30:])
    proto = np.concatenate(proto)
    query = np.concatenate(query)
    mu = H[proto].mean(0)
    Zp = (H[proto] - mu).astype(np.float64)
    yp0 = y[proto]
    yi = np.concatenate([np.where(yp0 == c)[0] for c in range(K)])
    Zc = [Zp[yp0 == c] for c in range(K)]

    # 每类 35 个副本，每个副本 15 张有放回
    reps, rc = [], []
    for c in range(K):
        pick = rng.randint(0, 30, (R, NSUB))
        reps.append(Zc[c][pick].mean(1))
        rc.append(np.full(R, c))
    P0 = np.concatenate(reps)
    rcls = np.concatenate(rc)

    qi = rng.choice(len(query), NQ, replace=False)
    Q = (H[query] - mu)[qi].astype(np.float64)
    qlab = y[query][qi]

    means = np.stack([Zc[c].mean(0) for c in range(K)])
    Sw = np.zeros((384, 384))
    for c in range(K):
        A = Zc[c] - means[c]
        Sw += A.T @ A
    Sb = np.zeros_like(Sw)
    for c in range(K):
        v = means[c][:, None]
        Sb += 30.0 * (v @ v.T)
    Sw /= len(Zp)
    Sb /= len(Zp)
    Sw += 1e-6 * np.trace(Sw) / 384 * np.eye(384)
    L = np.linalg.cholesky(Sw)
    Linv = np.linalg.inv(L)
    w, V = np.linalg.eigh(Linv @ Sb @ Linv.T)
    o = np.argsort(w)[::-1]
    V, w = V[:, o], w[o]

    for d in DIMS:
        Wd = Linv.T @ V[:, :d]
        P = P0 @ Wd
        Qp = Q @ Wd
        p2 = (P * P).sum(1)
        n = len(Qp)
        hit1 = np.zeros(n, bool)
        hit3 = np.zeros(n, bool)
        for i in range(0, n, BLK):
            q = Qp[i:i + BLK]
            dd = p2[None, :] - 2.0 * (q @ P.T) + (q * q).sum(1)[:, None]
            order = np.argsort(dd, axis=1)
            for r in range(len(q)):
                row = order[r]
                if rcls[row[0]] == qlab[i + r]:
                    hit1[i + r] = True
                seen = []
                for j in row[:400]:
                    c = rcls[j]
                    if c not in seen:
                        seen.append(c)
                        if len(seen) == 3:
                            break
                if qlab[i + r] in seen:
                    hit3[i + r] = True
        res[d]["a1"].append(float(hit1.mean()))
        res[d]["a3"].append(float(hit3.mean()))
        log(f"seed {seed} d={d}: A*_1={hit1.mean():.4f} 召回@3={hit3.mean():.4f}")

log("")
log(f"{'d':>4} {'A*_1':>8} {'召回@3':>8}")
for d in DIMS:
    log(f"{d:>4} {np.mean(res[d]['a1']):>8.4f} {np.mean(res[d]['a3']):>8.4f}")
json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=1)
log("已写", OUT)
