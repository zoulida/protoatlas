"""exp39: 没有任何节点拥有全局数据时，LDA 还能不能算？

问题：判别分析需要类内与类间散度，若没有节点持有全局数据，这两项是否根本无法计算？

本实验要澄清的核心：S_W 与 S_B 只需**相加**，不需要任何节点拥有全部数据。
要验证的是三个更实际的问题：
  A. 覆盖率：只有 k / 10 个类的统计量参与拟合，W 还能服务全部 10 类吗？
  B. 通信量：先在公共 PCA 子空间降到 D'=64/128，再算 LDA，精度掉多少？
  C. 局部方向拼接：每节点只有 2 类 → 本地 LDA 只有 1 个判别方向。把 20 个节点的
     rank-1 方向正交化拼起来，能恢复全局 LDA 多少？（这条大概率失败，失败本身也是结论）
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)
from exp37_metric_learn import fit_pca, fit_lda, proto_route, lin_probe


def log(*a, **kw):
    print(*a, flush=True, **kw)


def lda_stats(H, y):
    """节点本地能算出的全部统计量：(n_c, m_c, S_local 合计)"""
    mu = H.mean(0).astype(np.float64)
    X = H.astype(np.float64) - mu
    cls = np.unique(y)
    per = {}
    Sw_local = np.zeros((H.shape[1], H.shape[1]))
    for c in cls:
        Xc = X[y == c]
        mc = Xc.mean(0)
        Sw_local += Xc.T @ Xc - len(Xc) * np.outer(mc, mc)
        per[int(c)] = (len(Xc), Xc.mean(0) + mu)      # 局部均值（回到原坐标）
    return per, Sw_local, mu, len(X)


def merge_stats(stats):
    """模拟聚合者对若干节点的局部统计量求和（这就是 federated 的全部操作）"""
    tot_n = 0
    acc_mu = None
    acc_Sw = None
    per_sum = {}
    for per, Sw, mu_n, n_n in stats:
        tot_n += n_n
        acc_mu = mu_n * n_n if acc_mu is None else acc_mu + mu_n * n_n
        acc_Sw = Sw if acc_Sw is None else acc_Sw + Sw
        for c, (nc, mc) in per.items():
            s = per_sum.get(c, (0, np.zeros_like(mc)))
            per_sum[c] = (s[0] + nc, s[1] + nc * mc)
    mu = acc_mu / tot_n
    # 全局类均值
    gm = {c: (v[1] / v[0], v[0]) for c, v in per_sum.items()}
    # 精确校正：S_W^global = Σ_n [S_n + n_nc (m_nc - m_c)(m_nc - m_c)^T]
    Sw = acc_Sw.copy()
    for per, Sw_n, mu_n, n_n in stats:
        for c, (nc, mc) in per.items():
            if c not in gm:
                continue
            d = mc - gm[c][0]
            Sw += nc * np.outer(d, d)
    D = len(mu)
    K = len(gm)
    Sw /= max(1, tot_n - K)
    # 类间散度
    Sb = np.zeros((D, D))
    for c, (mc, nc) in gm.items():
        d = mc - mu
        Sb += nc * np.outer(d, d)
    Sb /= max(1, K - 1)
    return Sw, Sb, gm, mu, tot_n


def lda_from_stats(Sw, Sb, d, lam=1e-3):
    D = Sw.shape[0]
    tr = np.trace(Sw) / D
    Sw_r = Sw + lam * tr * np.eye(D)
    L = np.linalg.cholesky(Sw_r)
    Li = np.linalg.inv(L)
    A = Li @ Sb @ Li.T
    A = (A + A.T) / 2
    w, U = np.linalg.eigh(A)
    d_eff = min(d, U.shape[1])
    U = U[:, ::-1][:, :d_eff]
    W = Li.T @ U
    return W.astype(np.float32)


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    y_tr = ytr[tr_sel].astype(np.int64)
    y_te = yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)

    # 原型
    protos, pcls = [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
    P = np.stack(protos).astype(np.float32)
    pcls = np.array(pcls)

    mu0 = H.mean(0)
    G = lin_probe(H - mu0, y_tr, T - mu0, y_te, 10)
    log("=" * 88)
    log(f"CIFAR-10，{len(H)} 训练 / {len(T)} 测试，{Nn} 节点 × 2 类，φ=DINOv2 384")
    log(f"  集中式 G（全维线性头）= {G:.4f}")
    log(f"  全局 LDA-9 基准：", end="")
    mu, Wg, _ = fit_lda(H, y_tr, 12, whiten=True)
    r = proto_route((T - mu) @ Wg, (P - mu) @ Wg, pcls, y_te)
    log(f"R@1 = {r['R1']:.4f}  R@3 = {r['R3']:.4f}  (d={Wg.shape[1]})")

    res = {"G": G, "global": r}

    # ---------- A. 覆盖率扫描 ----------
    log("")
    log("A. 只有部分类的统计量参与拟合，W 还要服务全部 10 类")
    log(f"   {'参与类数':>9}{'聚合节点数':>11}{'d':>5}{'R@1':>9}{'R@3':>9}{'方向余弦':>10}")
    log("   " + "-" * 55)
    rng = np.random.RandomState(0)
    rows = []
    for k in [2, 4, 6, 8, 10]:
        sub = list(range(k))
        mask = np.isin(y_tr, sub)          # 只保留这 k 类的样本
        # 这些样本分布在若干节点上（节点的类若不在 sub 内则整节点没有贡献）
        nodes_used = [n for n in range(Nn) if set(cls_of_node[n]) & set(sub)]
        stats = []
        for n in nodes_used:
            m = mask & (tr_node == n)
            if m.sum() == 0:
                continue
            labs = y_tr[m]
            # 节点只用自己有的、且在 sub 内的类
            stats.append(lda_stats(H[m], labs))
        Sw, Sb, gm, mu_g, tot = merge_stats(stats)
        Wk = lda_from_stats(Sw, Sb, 12)
        r = proto_route((T - mu_g) @ Wk, (P - mu_g) @ Wk, pcls, y_te)
        # 与全局 W 的前 3 个方向比余弦（取最小绝对值，方向符号无关）
        Wg3 = Wg[:, :3]
        A = Wk[:, :min(3, Wk.shape[1])]
        cos = A.T @ Wg3
        cos = np.abs(cos).max(1).mean()
        log(f"   {k:>9}{len(nodes_used):>11}{Wk.shape[1]:>5}{r['R1']:>9.4f}"
            f"{r['R3']:>9.4f}{cos:>10.4f}")
        rows.append(dict(k=k, nodes=len(nodes_used), d=int(Wk.shape[1]),
                         **r, cos=float(cos)))
    res["coverage"] = rows

    # ---------- B. 先在公共 PCA 子空间降维再算 LDA ----------
    log("")
    log("B. 先在公共 PCA 子空间降到 D'，再算 LDA（目的是压小 S_W 的通信量）")
    log(f"   {'D_pca':>7}{'Sw 体积':>11}{'d':>5}{'R@1':>9}{'R@3':>9}{'N=1000 聚合量':>14}")
    log("   " + "-" * 55)
    rowsB = []
    for Dp in [32, 64, 128, 384]:
        if Dp == 384:
            Hp, Tp, Pp = H, T, P
            mu_p = np.zeros(384, np.float32)
        else:
            mu_p, Wp = fit_pca(H, Dp)
            Hp, Tp, Pp = (H - mu_p) @ Wp, (T - mu_p) @ Wp, (P - mu_p) @ Wp
        Sw, Sb, _, _, _ = merge_stats([lda_stats(Hp, y_tr)])
        Wk = lda_from_stats(Sw, Sb, 12)
        mu_z = Hp.mean(0)
        r = proto_route((Tp - mu_z) @ Wk, (Pp - mu_z) @ Wk, pcls, y_te)
        kb = Dp * Dp * 4 / 1024
        log(f"   {Dp:>7}{kb:>10.0f}K{Wk.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}"
            f"{kb*1000/1024/1024:>12.0f}MB")
        rowsB.append(dict(Dp=Dp, sw_kb=kb, d=int(Wk.shape[1]), **r))
    res["subspace"] = rowsB

    # ---------- C. 各节点本地 rank-1 方向拼接 ----------
    log("")
    log("C. 每节点只有 2 类 → 本地 LDA 只有 1 个方向。把 20 个 rank-1 方向正交化拼接")
    dirs = []
    for n in range(Nn):
        m = tr_node == n
        _, Wn, _ = fit_lda(H[m], y_tr[m], 4, whiten=True)
        if Wn.shape[1] > 0:
            v = Wn[:, 0]
            dirs.append(v / (np.linalg.norm(v) + 1e-12))
    Dm = np.stack(dirs, 1)                       # (384, 20)
    # Gram-Schmidt 正交化
    Q, R = np.linalg.qr(Dm)
    for d_use in [3, 5, 9]:
        Wc = Q[:, :d_use].astype(np.float32)
        mu_c = H.mean(0)
        r = proto_route((T - mu_c) @ Wc, (P - mu_c) @ Wc, pcls, y_te)
        log(f"   拼接 d={d_use:<3}  R@1 = {r['R1']:.4f}  R@3 = {r['R3']:.4f}")
        res.setdefault("concat", []).append(dict(d=d_use, **r))

    with open(os.path.join(ROOT, "results", "results_fedlda.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n已保存 results_fedlda.json")


def _prep(*a):
    return None


if __name__ == "__main__":
    main()
