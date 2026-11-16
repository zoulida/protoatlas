"""exp37: 投影层该不该"训练"？—— PCA(无监督) vs Fisher/LDA(有监督) vs 全维马氏

背景：
  exp32 的结构性不等式：R > G/S 才赢。R 目前用 PCA-16 坐标，只有 0.9422，
  门槛 G/S = 0.970，差 2.8 点。
  用户的问法："能不能训练一个模型，让它在空间里把类划开？"——这正是度量学习。

本实验只动一件事：坐标投影 W 怎么来。φ 固定不动（冻结 DINOv2）。
  1. PCA-d        无监督，方差最大
  2. LDA-d        有监督，Fisher 判据（类间/类内）最大化，并按类内协方差白化
  3. LDA-dir-d    有监督，但只要方向不要白化
  4. 全维马氏     不压缩，直接用 S_W^{-1} 做最近类均值（NCM）

指标：
  R     路由命中（最近原型类 == 真类）
  R@3   前三候选中含真类（按类去重后）
  G_ful 集中式（全维 384 线性头）—— 不动，作为门槛的分母来源
  G_d   同一个 d 维空间里的线性头（看投影丢了多少分类信息）
  S     专家（全维，各节点自己那 2 类）

LDA 的代价必须说清：它需要**全部类的标签**（全局监督），PCA 不需要。
但 LDA 需要的只是"类内散度矩阵 + 类均值"，两者都可加和 → 联邦环境里可由
各节点各自算 384x384 矩阵后 gossip 求和，一轮搞定，见文末讨论。
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


# ---------------- 投影构造 ----------------

def fit_pca(H, d):
    mu = H.mean(0)
    Xc = H - mu
    C = (Xc.T @ Xc) / (len(H) - 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    return mu, V[:, ::-1][:, :d].astype(np.float32)


def fit_lda(H, y, d, lam=1e-3, whiten=True):
    """返回 mu, W(d 列)。whiten=True 时 W^T S_W W = I（Fisher 白化）。"""
    mu = H.mean(0).astype(np.float64)
    X = H.astype(np.float64) - mu
    cls = np.unique(y)
    K, D = len(cls), H.shape[1]

    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in cls:
        m = y == c
        Xc = X[m]
        mc = Xc.mean(0)
        Sw += Xc.T @ Xc - len(Xc) * np.outer(mc, mc)
        Sb += len(Xc) * np.outer(mc, mc)
    Sw /= max(1, len(X) - K)
    Sb /= max(1, K - 1)

    # 正则化，避免 Sw 奇异
    tr = np.trace(Sw) / D
    Sw_r = Sw + lam * tr * np.eye(D)

    # 白化：A = Sw^{-1/2} Sb Sw^{-1/2} 对称化后取特征
    L = np.linalg.cholesky(Sw_r)                    # Sw_r = L L^T
    Li = np.linalg.inv(L)
    A = Li @ Sb @ Li.T
    A = (A + A.T) / 2
    w, U = np.linalg.eigh(A)
    d_eff = min(d, K - 1)                            # LDA 秩 ≤ K-1
    U = U[:, ::-1][:, :d_eff]
    w = w[::-1][:d_eff]

    if whiten:
        W = Li.T @ U                                 # 使 W^T Sw W = I
    else:
        W = Li.T @ U
        # 只要方向：把每个方向归一化，不做类内等方差
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
        # 再乘回 sqrt(eigenvalue) 让"判别力"体现在尺度上
        W = W * np.sqrt(np.maximum(w, 0))[None, :]
    return mu, W.astype(np.float32), w


def fit_full_maha(H, y, lam=1e-3):
    """全维马氏：不降维，直接白化。d=D"""
    mu, W, w = fit_lda(H, y, H.shape[1], lam=lam, whiten=True)
    # LDA 只给 K-1 个判别方向；剩下的用 Sw 白化后的残余方向补齐到 D 维
    D = H.shape[1]
    if W.shape[1] < D:
        X = H.astype(np.float64) - mu
        Sw = np.zeros((D, D))
        for c in np.unique(y):
            Xc = X[y == c]
            mc = Xc.mean(0)
            Sw += Xc.T @ Xc - len(Xc) * np.outer(mc, mc)
        Sw /= max(1, len(X) - len(np.unique(y)))
        Sw += lam * (np.trace(Sw) / D) * np.eye(D)
        Li = np.linalg.inv(np.linalg.cholesky(Sw))
        # 补齐方向：白化空间的单位阵（与已有判别方向正交补）
        C = Li.T
        Q, _ = np.linalg.qr(np.hstack([W, C]))
        W = Q[:, :D]
    return mu, W.astype(np.float32)


# ---------------- 评估 ----------------

def proto_route(Zq, Zp, pcls, yte, ks=(1, 3)):
    D2 = (Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2 * Zq @ Zp.T
    order = np.argsort(D2, 1)
    out = {}
    hit1 = float(np.mean(pcls[order[:, 0]] == yte))
    out["R1"] = hit1
    # 按类去重后取前 k 个不同类
    for k in ks[1:]:
        ok = np.zeros(len(yte), bool)
        for i in range(len(yte)):
            seen, got = set(), []
            for j in order[i]:
                c = int(pcls[j])
                if c in seen:
                    continue
                seen.add(c)
                got.append(c)
                if len(got) >= k:
                    break
            ok[i] = yte[i] in got
        out[f"R{k}"] = float(ok.mean())
    return out


def lin_probe(Z, y, Zt, yt, n_cls, lam=1e-2):
    """岭回归 one-hot（闭式），比 torch 训练快且稳"""
    n, d = Z.shape
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), y] = 1
    A = Z.T @ Z + lam * n * np.eye(d)
    B = Z.T @ Y
    Wt = np.linalg.solve(A, B)
    return float(np.mean((Zt @ Wt).argmax(1) == yt))


def main():
    from exp12_proto_route import build_split
    import pandas as pd

    res = {}

    # ================= CIFAR-10：决定性测试 =================
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    y_tr = ytr[tr_sel].astype(np.int64)
    y_te = yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)

    log("=" * 84)
    log(f"=== CIFAR-10（{len(H)} 训练 / {len(T)} 测试，20 节点 × 2 类）===")
    log(f"  φ 冻结：DINOv2 ViT-S/14，{H.shape[1]} 维")

    # 原型（按节点收集，与 exp32 同口径）
    protos, pcls = [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
    P = np.stack(protos).astype(np.float32)
    pcls = np.array(pcls)

    # G / S 不随投影变化（都在全维上算）
    mu0 = H.mean(0)
    G_full = lin_probe(H - mu0, y_tr, T - mu0, y_te, 10)
    S_list = []
    for n in range(Nn):
        m = tr_node == n
        lab = np.searchsorted(cls_of_node[n], y_tr[m])
        Wt = np.zeros((H.shape[1], 2))
        Z = H[m] - mu0
        Y = np.zeros((m.sum(), 2))
        Y[np.arange(m.sum()), lab] = 1
        A = Z.T @ Z + 1e-2 * len(Z) * np.eye(Z.shape[1])
        Wt = np.linalg.solve(A, Z.T @ Y)
        idx = np.where(np.isin(y_te, cls_of_node[n]))[0]
        pr = (T[idx] - mu0) @ Wt
        S_list.append(float(np.mean(np.array(cls_of_node[n])[pr.argmax(1)] == y_te[idx])))
    S = float(np.mean(S_list))
    thr = G_full / S
    log(f"  G 集中式（全维 384 线性头）  {G_full:.4f}")
    log(f"  S 专家（全维 384，各节点 2 类）{S:.4f}")
    log(f"  → 路由门槛 G/S = {thr:.4f}   （R 必须超过它才赢）")
    log("")
    log(f"  {'投影方法':<14}{'d':>5}{'R@1':>9}{'R@3':>9}{'G_d':>9}{'过门槛':>8}")
    log("  " + "-" * 62)

    rows10 = []
    for d in [8, 16, 32, 64, 128]:
        # PCA
        mu, W = fit_pca(H, d)
        r = proto_route((T - mu) @ W, (P - mu) @ W, pcls, y_te)
        gd = lin_probe((H - mu) @ W, y_tr, (T - mu) @ W, y_te, 10)
        log(f"  {'PCA':<14}{d:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}{gd:>9.4f}"
            f"{'赢' if r['R1'] > thr else '输':>8}")
        rows10.append(dict(method="PCA", d=d, **r, G_d=gd))

        # LDA
        mu, W, _ = fit_lda(H, y_tr, d, whiten=True)
        r = proto_route((T - mu) @ W, (P - mu) @ W, pcls, y_te)
        gd = lin_probe((H - mu) @ W, y_tr, (T - mu) @ W, y_te, 10)
        log(f"  {'LDA(Fisher)':<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}{gd:>9.4f}"
            f"{'赢' if r['R1'] > thr else '输':>8}")
        rows10.append(dict(method="LDA", d=int(W.shape[1]), **r, G_d=gd))

        # LDA 方向（不白化）
        mu, W, _ = fit_lda(H, y_tr, d, whiten=False)
        r = proto_route((T - mu) @ W, (P - mu) @ W, pcls, y_te)
        log(f"  {'LDA-dir':<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}"
            f"{'—':>9}{'赢' if r['R1'] > thr else '输':>8}")
        rows10.append(dict(method="LDA-dir", d=int(W.shape[1]), **r))

    # 全维马氏
    mu, W = fit_full_maha(H, y_tr)
    r = proto_route((T - mu) @ W, (P - mu) @ W, pcls, y_te)
    log(f"  {'全维马氏':<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}"
        f"{'—':>9}{'赢' if r['R1'] > thr else '输':>8}")
    rows10.append(dict(method="Mahalanobis", d=int(W.shape[1]), **r))
    res["cifar10"] = dict(G=G_full, S=S, thr=thr, rows=rows10,
                          pca16_baseline=0.9422, note="PCA-16 baseline from exp32")

    # ================= CIFAR-100：K 变大时投影方法差多少 =================
    log("")
    log("=" * 84)
    log("=== CIFAR-100（500/类 训练，30/类 测试，100 类）===")
    H1 = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    T1 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    y1 = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int64)
    yt1 = np.load(os.path.join(CACHE, "c100.npz"))["y"].astype(np.int64)
    log(f"  训练 {H1.shape} / 测试 {T1.shape}")

    protos1, pcls1 = [], []
    for c in range(100):
        protos1.append(H1[y1 == c].mean(0))
        pcls1.append(c)
    P1 = np.stack(protos1).astype(np.float32)
    pcls1 = np.array(pcls1)

    mu0 = H1.mean(0)
    G1 = lin_probe(H1 - mu0, y1, T1 - mu0, yt1, 100)
    log(f"  G 集中式（全维 384 线性头）  {G1:.4f}")
    log("")
    log(f"  {'投影方法':<14}{'d':>5}{'R@1':>9}{'R@3':>9}{'G_d':>9}{'解释方差':>10}")
    log("  " + "-" * 57)

    rows100 = []
    for d in [16, 32, 64, 128]:
        mu, W = fit_pca(H1, d)
        r = proto_route((T1 - mu) @ W, (P1 - mu) @ W, pcls1, yt1)
        gd = lin_probe((H1 - mu) @ W, y1, (T1 - mu) @ W, yt1, 100)
        log(f"  {'PCA':<14}{d:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}{gd:>9.4f}{'—':>10}")
        rows100.append(dict(method="PCA", d=d, **r, G_d=gd))

        mu, W, _ = fit_lda(H1, y1, d, whiten=True)
        r = proto_route((T1 - mu) @ W, (P1 - mu) @ W, pcls1, yt1)
        gd = lin_probe((H1 - mu) @ W, y1, (T1 - mu) @ W, yt1, 100)
        log(f"  {'LDA(Fisher)':<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}{gd:>9.4f}{'—':>10}")
        rows100.append(dict(method="LDA", d=int(W.shape[1]), **r, G_d=gd))

    mu, W = fit_full_maha(H1, y1)
    r = proto_route((T1 - mu) @ W, (P1 - mu) @ W, pcls1, yt1)
    gd = lin_probe((H1 - mu) @ W, y1, (T1 - mu) @ W, yt1, 100)
    log(f"  {'全维马氏':<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}{gd:>9.4f}{'—':>10}")
    rows100.append(dict(method="Mahalanobis", d=int(W.shape[1]), **r, G_d=gd))
    res["cifar100"] = dict(G=G1, rows=rows100, knn_full_dim=0.8093, linear_probe_full=0.8230)

    with open(os.path.join(ROOT, "results", "results_metric_learn.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n已保存 results_metric_learn.json")


if __name__ == "__main__":
    main()
