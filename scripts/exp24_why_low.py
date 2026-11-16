"""exp24: 0.5647 到底是不是"基本不成功"？

三个对照把责任分清：
  ① 全维 384 维的 1-NN（不用 PCA）—— 若也是 0.57，说明降维没损失
  ② 线性探针（ridge 闭式解，有监督）—— 这是"特征本身能到的上界"
  ③ 正确类的排名分布 —— 43.5% 的错误里，多少是"差一点"，多少是真找不到

CIFAR-10 同口径并列，作为"简单任务"的参照。
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


def fit_pca(H, k):
    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    return mu[0], V[:, -k:].astype(np.float32)


def knn1(Q, G, chunk=256):
    """返回每个 query 在 G 中最近的下标与距离"""
    gn = (G ** 2).sum(1)
    idx = np.empty(len(Q), np.int64)
    for i in range(0, len(Q), chunk):
        a = (Q[i:i + chunk] ** 2).sum(1)[:, None] + gn[None, :] - 2.0 * Q[i:i + chunk] @ G.T
        idx[i:i + chunk] = np.argmin(a, 1)
    return idx


def ridge_probe(H, y, n_cls, lam=1.0):
    """闭式 ridge：W = (X^T X + lam I)^-1 X^T Y"""
    X = H.astype(np.float64)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.zeros((len(X), n_cls), np.float64)
    Y[np.arange(len(X)), y] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ Y)
    return W


def main():
    import pandas as pd
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H10 = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T10 = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y10, yT10 = ytr[tr_sel].astype(np.int32), yte[te_sel].astype(np.int32)

    H100 = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    y100 = d["y"].astype(np.int32)
    T100 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    yT100 = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:3000]

    res = {}
    for tag, H, y, T, yT, n_cls in [
            ("CIFAR-10", H10, y10, T10, yT10, 10),
            ("CIFAR-100", H100, y100, T100, yT100, 100)]:
        log("\n" + "=" * 72)
        log(f"=== {tag} === 训练 {H.shape} 查询 {len(T)} 类 {n_cls}")
        log(f"  随机猜测基线 1/{n_cls} = {1/n_cls:.4f}")

        row = {}
        # ① 单样本 1-NN：全维 vs PCA-16
        idx384 = knn1(T, H)
        acc384 = float(np.mean(y[idx384] == yT))
        mu, W = fit_pca(H, 16)
        idx16 = knn1((T - mu) @ W, (H - mu) @ W)
        acc16 = float(np.mean(y[idx16] == yT))
        log(f"  单样本 1-NN  全维384: {acc384:.4f}   PCA-16: {acc16:.4f}"
            f"   降维损失 {acc384-acc16:+.4f}")
        row.update(knn1_384=acc384, knn1_pca16=acc16)

        # ② 原型级 1-NN（每类 500 张均值）
        protos = np.stack([H[y == c].mean(0) for c in range(n_cls)]).astype(np.float32)
        p_acc384 = float(np.mean(yT[idx384[:0]] if False else
                                 np.array(range(n_cls))[knn1(T, protos)] == yT))
        p_acc16 = float(np.mean(np.array(range(n_cls))[knn1((T - mu) @ W,
                                                          (protos - mu) @ W)] == yT))
        log(f"  原型级 1-NN  全维384: {p_acc384:.4f}   PCA-16: {p_acc16:.4f}")
        row.update(proto1_384=p_acc384, proto1_pca16=p_acc16)

        # ③ 线性探针（有监督上界）
        Wp = ridge_probe(H, y, n_cls)
        Xq = np.hstack([T.astype(np.float64), np.ones((len(T), 1))])
        pred = np.argmax(Xq @ Wp, 1)
        probe = float(np.mean(pred == yT))
        log(f"  线性探针（ridge）: {probe:.4f}   ← 特征的可分性上界")
        row["ridge_probe"] = probe

        # ④ 正确类的排名分布（原型级 PCA-16，按类去重）
        mu2, W2 = fit_pca(H, 16)
        Zq, Zp = (T - mu2) @ W2, (protos - mu2) @ W2
        D = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
        order = np.argsort(D, 1)
        rank = np.array([int(np.where(order[i] == yT[i])[0][0]) + 1 for i in range(len(T))])
        log("  正确类的原始排名分布（未去重）：")
        bins = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 6)]
        for a, b in bins:
            m = (rank >= a) & (rank <= b)
            lab = f"{a}" if a == b else (f"{a}-{b}" if b < 10 ** 6 else f">{a-1}")
            log(f"    第 {lab:<8} 名  {m.mean():.4f}")
            row[f"rank_{lab}"] = float(m.mean())
        log(f"  前 10 名覆盖率 {float((rank<=10).mean()):.4f}；前 20 名 {float((rank<=20).mean()):.4f}")

        res[tag] = row

    with open(os.path.join(ROOT, "results", "results_why_low.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
