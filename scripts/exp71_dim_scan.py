"""exp71: 固定特征、变投影维数，看路由上限 A*_1 随 d 的变化

问题（作者）：1000 类时 384 维不够用，换成 700 多维能不能提高？
做法：不换特征，只在同一份 384 维 DINOv2 特征上，保留判别分析的前 d 个方向，
      d 从 8 扫到 384，看最近原型命中率 A*_1 随 d 怎么走。
判读：
  - 若 d 到 384 仍在明显上升 → 判别信息没被吃尽，加维有意义；
  - 若早就平了（甚至下降）→ 加维无益，瓶颈在特征的判别力，不在维数。
设定与 exp56 对齐：K=1000，每类 50 张，30 张作原型池、20 张作查询，抽 3000 条查询。
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
K = 1000
DIMS = [8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384]
SEEDS = (0, 1, 2)
NQ = 3000
BLK = 300
OUT = os.path.join(ROOT, "results", "results_dim_scan.json")


def log(*a):
    print(*a, flush=True)


def nearest_class(Zp, yp, Q, yq, dedup=3):
    """返回 A*_1（最近原型即真类）与召回@3（去重后）"""
    n = len(Q)
    hit1 = np.zeros(n, bool)
    hit3 = np.zeros(n, bool)
    p2 = (Zp * Zp).sum(1)
    for i in range(0, n, BLK):
        q = Q[i:i + BLK]
        d = p2[None, :] - 2.0 * (q @ Zp.T) + (q * q).sum(1)[:, None]
        order = np.argsort(d, axis=1, kind="quicksort")
        for r in range(len(q)):
            row = order[r]
            if yp[row[0]] == yq[i + r]:
                hit1[i + r] = True
            seen = []
            for j in row[:200]:
                c = yp[j]
                if c not in seen:
                    seen.append(c)
                    if len(seen) == dedup:
                        break
            if yq[i + r] in seen:
                hit3[i + r] = True
    return float(hit1.mean()), float(hit3.mean())


H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
y = np.load(os.path.join(CACHE, "in1k_y.npy"))
log("特征", H.shape)

res = {d: {"a1": [], "a3": [], "var": []} for d in DIMS}
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
    Zp = H[proto] - mu
    Zq = H[query] - mu
    yp = y[proto]
    qi = rng.choice(len(query), NQ, replace=False)
    Q = Zq[qi]
    qlab = y[query][qi]

    # 判别分析：类均值、类内散度、类间散度
    means = np.stack([Zp[yp == c].mean(0) for c in range(K)])
    Sw = np.zeros((Zp.shape[1], Zp.shape[1]), np.float64)
    for c in range(K):
        A = Zp[yp == c] - means[c]
        Sw += A.T @ A
    Sb = np.zeros_like(Sw)
    for c in range(K):
        v = means[c][:, None]
        Sb += 30.0 * (v @ v.T)
    Sw /= len(Zp)
    Sb /= len(Zp)
    Sw += 1e-6 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    L = np.linalg.cholesky(Sw)
    Linv = np.linalg.inv(L)
    M = Linv @ Sb @ Linv.T
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    tot = w.clip(min=0).sum()
    log(f"seed {seed}: 前 d 个方向累计解释 {[(d, round(float(w[:d].clip(min=0).sum()/tot),3)) for d in (32,64,128,256,384)]}")

    for d in DIMS:
        Wd = Linv.T @ V[:, :d]            # D x d
        P = Zp @ Wd
        Qp = Q @ Wd
        a1, a3 = nearest_class(P, yp, Qp, qlab)
        res[d]["a1"].append(a1)
        res[d]["a3"].append(a3)
        res[d]["var"].append(float(w[:d].clip(min=0).sum() / tot))
    del Sw, Sb, M

log("")
log(f"{'d':>4} {'解释比例':>8} {'A*_1':>8} {'±':>6} {'召回@3':>8}")
for d in DIMS:
    a1 = np.array(res[d]["a1"])
    a3 = np.array(res[d]["a3"])
    log(f"{d:>4} {np.mean(res[d]['var']):>8.3f} {a1.mean():>8.4f} {a1.std():>6.4f} {a3.mean():>8.4f}")

json.dump({str(k): {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in res.items()},
          open(OUT, "w"), indent=1)
log("已写", OUT)
