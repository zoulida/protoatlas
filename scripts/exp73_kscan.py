"""exp73: 类数增加时，各类在坐标空间里的区分度怎么变

背景（作者担心）：随着类型变多、节点变多，各类的区分度可能不够，想重训特征提取器。

本脚本固定「每类样本数」，只让类数 K 变，看最近原型精度与类间余量怎么走。
ImageNet-1k 特征缓存（5 万张、1000 类、每类 50 张）：每类 30 张作原型、20 张作查询。

  K = 10, 20, 50, 100, 200, 500, 1000（每档随机抽同样多的类，5 个种子）

读三组数：
  A1        最近原型路由精度（按类去重）
  margin    最近的异类原型距离 / 类内半径 的中位数（越大越好分）
  d         该档实际用到的投影维数 min(K-1, 384)
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
KS = [10, 20, 50, 100, 200, 500, 1000]
SEEDS = 5


def log(*a):
    print(*a, flush=True)


def run(H, y, K, seed):
    rng = np.random.default_rng(seed)
    cls = rng.choice(1000, K, replace=False)
    Hp, yp, Hq, yq = [], [], [], []
    for c in cls:
        X = H[y == c]
        if len(X) < N_PROTO + N_QUERY:
            return None
        Hp.append(X[:N_PROTO])
        Hq.append(X[N_PROTO:N_PROTO + N_QUERY])
        yp.append(np.full(N_PROTO, c))
        yq.append(np.full(N_QUERY, c))
    Hp = np.concatenate(Hp).astype(np.float32)
    Hq = np.concatenate(Hq).astype(np.float32)
    yp = np.concatenate(yp)
    yq = np.concatenate(yq)

    mu, W, d = fit_lda(Hp, yp, min(K - 1, Hp.shape[1]))
    Zp = (Hp - mu) @ W
    Zq = (Hq - mu) @ W

    protos, pcls, spread = [], [], []
    for c in cls:
        Zc = Zp[yp == c]
        protos.append(Zc.mean(0))
        pcls.append(c)
        spread.append(np.sqrt(((Zc - Zc.mean(0)) ** 2).sum(1)).mean())
    P = np.stack(protos).astype(np.float64)
    pcls = np.array(pcls)
    spread = float(np.median(spread))

    Dm = np.sqrt(np.maximum((Zq.astype(np.float64) ** 2).sum(1)[:, None]
                            + (P ** 2).sum(1)[None, :]
                            - 2 * Zq.astype(np.float64) @ P.T, 0))

    # 最近原型（按类去重：同类只算一次）
    order = np.argsort(Dm, 1)
    pred = np.zeros(len(Zq), np.int64)
    for i in range(len(Zq)):
        seen = set()
        for j in order[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.add(c)
            pred[i] = c
            break
    a1 = float(np.mean(pred == yq))

    # 类间余量：每个原型到最近的异类原型的距离
    Pd = np.sqrt(np.maximum((P ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :]
                            - 2 * P @ P.T, 0))
    np.fill_diagonal(Pd, np.inf)
    margin = float(np.median(Pd.min(1)) / max(spread, 1e-9))

    return dict(K=K, d=int(d), a1=a1, margin=margin, spread=spread)


def main():
    H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "in1k_y.npy"))
    log(f"特征 {H.shape}，类数 {len(np.unique(y))}")

    res = []
    for K in KS:
        rows = [run(H, y, K, s) for s in range(SEEDS)]
        rows = [r for r in rows if r]
        a1 = float(np.mean([r["a1"] for r in rows]))
        a1s = float(np.std([r["a1"] for r in rows]))
        mg = float(np.mean([r["margin"] for r in rows]))
        d = rows[0]["d"]
        res.append(dict(K=K, d=d, a1=a1, a1_std=a1s, margin=mg))
        log(f"  K={K:5d}  d={d:4d}  A1 {a1:.4f} ±{a1s:.4f}  margin {mg:6.2f}")

    with open(os.path.join(ROOT, "results", "results_kscan.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_kscan.json")


if __name__ == "__main__":
    main()
