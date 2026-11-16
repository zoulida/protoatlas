"""exp74: 类数多的时候，区分度不足是「类挤在一起」还是「每类样本太少」

K 固定为 1000（以及 100 作对照），只改用来算原型的样本数 k。
ImageNet-1k 缓存每类 50 张：k 张作原型，其余作查询。

若 A1 随 k 明显上升 -> 主因是原型估计噪声，加样本就能救，不必重训 φ。
若 A1 随 k 基本不动 -> 主因是类在空间里挤，噪声已不是瓶颈，才轮到更强的特征。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp32_end2end_all_lda import fit_lda  # noqa: E402

KS = [1000, 100]
KSAMP = [5, 10, 15, 20, 30, 40]
SEEDS = 3


def log(*a):
    print(*a, flush=True)


def run(H, y, K, k, seed):
    rng = np.random.default_rng(seed)
    cls = rng.choice(1000, K, replace=False)
    Hp, yp, Hq, yq = [], [], [], []
    for c in cls:
        X = H[y == c]
        idx = rng.permutation(len(X))
        Hp.append(X[idx[:k]])
        Hq.append(X[idx[k:]])
        yp.append(np.full(k, c))
        yq.append(np.full(len(X) - k, c))
    Hp, Hq = np.concatenate(Hp).astype(np.float32), np.concatenate(Hq).astype(np.float32)
    yp, yq = np.concatenate(yp), np.concatenate(yq)

    mu, W, d = fit_lda(Hp, yp, min(K - 1, Hp.shape[1]))
    Zp = (Hp - mu) @ W
    Zq = (Hq - mu) @ W

    P, pcls, spread = [], [], []
    for c in cls:
        Zc = Zp[yp == c]
        P.append(Zc.mean(0))
        pcls.append(c)
        spread.append(np.sqrt(((Zc - Zc.mean(0)) ** 2).sum(1)).mean())
    P = np.stack(P).astype(np.float64)
    pcls = np.array(pcls)
    spread = float(np.median(spread))

    Dm = np.sqrt(np.maximum((Zq.astype(np.float64) ** 2).sum(1)[:, None]
                            + (P ** 2).sum(1)[None, :]
                            - 2 * Zq.astype(np.float64) @ P.T, 0))
    pred = np.zeros(len(Zq), np.int64)
    order = np.argsort(Dm, 1)
    for i in range(len(Zq)):
        seen = set()
        for j in order[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.add(c)
            pred[i] = c
            break
    Pd = np.sqrt(np.maximum((P ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :]
                            - 2 * P @ P.T, 0))
    np.fill_diagonal(Pd, np.inf)
    return dict(a1=float(np.mean(pred == yq)),
                margin=float(np.median(Pd.min(1)) / max(spread, 1e-9)))


def main():
    H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "in1k_y.npy"))
    res = []
    for K in KS:
        for k in KSAMP:
            rows = [run(H, y, K, k, s) for s in range(SEEDS)]
            a1 = float(np.mean([r["a1"] for r in rows]))
            a1s = float(np.std([r["a1"] for r in rows]))
            mg = float(np.mean([r["margin"] for r in rows]))
            res.append(dict(K=K, k=k, a1=a1, a1_std=a1s, margin=mg))
            log(f"  K={K:5d}  原型样本 k={k:3d}  A1 {a1:.4f} ±{a1s:.4f}  margin {mg:6.2f}")
    with open(os.path.join(ROOT, "results", "results_ksamp.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_ksamp.json")


if __name__ == "__main__":
    main()
