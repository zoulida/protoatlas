"""exp38: 用 LDA 坐标重测 exp32 的弱特征组——"训练投影"能否让路由翻盘？

exp32 的结论是"弱特征不翻盘"，但那一轮的 R 用的是 PCA-16 坐标。
exp37 显示：同样维度下 LDA 比 PCA 高 2.5~4 点，而且 9 维 LDA 就无损保留了 10 类的
全部分类能力（G_d = G_full = 0.9692）。所以"弱特征下 R 崩到 0.274"很可能是
投影方式造成的，不是特征的必然。

本实验：三组数据 × 两种投影，统一口径重算 G / S / 门槛 / R / 端到端。
  组1  CIFAR-10 + DINOv2        （强特征，G 高，S-G 小）
  组2  CIFAR-10 + 原始像素 3072  （弱特征，G 低，S-G 极大）← exp32 的翻盘假设
  组3  CIFAR-100 + DINOv2        （类多，G 中等）
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)
from exp37_metric_learn import fit_pca, fit_lda, proto_route, lin_probe


def log(*a):
    print(*a, flush=True)


def ridge_binary(X, lab, n_cls, lam=1e-2):
    n, d = X.shape
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), lab] = 1
    A = X.T @ X + lam * n * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def run_group(tag, H, T, y_tr, y_te, nodes, cls_of_node, node_of_cls, K, dims):
    log("")
    log("=" * 88)
    nodes_u = np.unique(nodes)
    log(f"=== {tag} ===   φ 维 {H.shape[1]}，{K} 类，{len(nodes_u)} 个节点")
    mu0 = H.mean(0)
    Hc, Tc = H - mu0, T - mu0

    G = lin_probe(Hc, y_tr, Tc, y_te, K)

    # 专家 S：每个节点只分自己那些类，在"属于自己类"的测试样本上测
    S_list = []
    for n in nodes_u:
        m = nodes == n
        cls = cls_of_node[int(n)]
        lab = np.searchsorted(cls, y_tr[m])
        Wt = ridge_binary(Hc[m], lab, len(cls))
        idx = np.where(np.isin(y_te, cls))[0]
        if len(idx) == 0:
            continue
        pr = np.array(cls)[(Tc[idx] @ Wt).argmax(1)]
        S_list.append(float(np.mean(pr == y_te[idx])))
    S = float(np.mean(S_list))
    thr = G / S

    log(f"  G 集中式（全维线性头）  {G:.4f}")
    log(f"  S 专家（全维，自家类）  {S:.4f}   S-G = {(S-G)*100:+.1f} 点")
    log(f"  → 门槛 G/S = {thr:.4f}（R 必须超过它）")
    log("")
    log(f"  {'投影':<14}{'d':>5}{'R@1':>9}{'R@3':>9}{'R×S':>9}{'vs G':>9}{'结论':>7}")
    log("  " + "-" * 62)

    # 原型
    protos, pcls, pnode = [], [], []
    for n in np.unique(nodes):
        for c in cls_of_node[int(n)]:
            protos.append(H[(nodes == n) & (y_tr == c)].mean(0))
            pcls.append(c)
            pnode.append(int(n))
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)

    rows = []
    for method in ["PCA", "LDA"]:
        for d in dims:
            if method == "PCA":
                mu, W = fit_pca(H, d)
            else:
                mu, W, _ = fit_lda(H, y_tr, d, whiten=True)
            r = proto_route((T - mu) @ W, (P - mu) @ W, pcls, y_te)
            RS = r["R1"] * S
            diff = RS - G
            log(f"  {method:<14}{W.shape[1]:>5}{r['R1']:>9.4f}{r['R3']:>9.4f}"
                f"{RS:>9.4f}{diff*100:>+8.1f}{'赢' if diff > 0 else '输':>7}")
            rows.append(dict(method=method, d=int(W.shape[1]), **r, RS=RS, vs_G=diff))
    return dict(tag=tag, G=G, S=S, thr=thr, rows=rows)


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    y_tr = ytr[tr_sel].astype(np.int64)
    y_te = yte[te_sel].astype(np.int64)
    cls_of_node = {n: sorted(nc[n]) for n in range(C["n_nodes"])}
    node_of_cls = {c: n for n in cls_of_node for c in cls_of_node[n]}

    res = {}

    Hd = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    Td = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    res["c10_dino"] = run_group("组1  CIFAR-10 + DINOv2（强特征）", Hd, Td, y_tr, y_te,
                                tr_node, cls_of_node, node_of_cls, 10, [8, 16, 32])

    d10 = np.load(os.path.join(CACHE, "cifar10.npz"))
    Xr = d10["xtr"][tr_sel].reshape(-1, 3072).astype(np.float32) / 255.0
    Xe = d10["xte"][te_sel].reshape(-1, 3072).astype(np.float32) / 255.0
    m_, s_ = Xr.mean(0, keepdims=True), Xr.std(0, keepdims=True) + 1e-6
    Xr, Xe = (Xr - m_) / s_, (Xe - m_) / s_
    # 3072 维直接算 LDA 太重，先 PCA-512（不做白化）再交给 LDA；PCA 组也用同一 512 维底
    mu_p, W_p = fit_pca(Xr, 512)
    Xr512, Xe512 = (Xr - mu_p) @ W_p, (Xe - mu_p) @ W_p
    res["c10_raw"] = run_group("组2  CIFAR-10 + 原始像素（弱特征，先 PCA-512 保底）",
                               Xr512, Xe512, y_tr, y_te, tr_node, cls_of_node,
                               node_of_cls, 10, [16, 32, 64, 128])

    H1 = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    T1 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    y1 = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int64)
    yt1 = np.load(os.path.join(CACHE, "c100.npz"))["y"].astype(np.int64)
    cls1 = {c: [c] for c in range(100)}
    nodes1 = y1.copy()
    res["c100_dino"] = run_group("组3  CIFAR-100 + DINOv2（100 类 × 1 节点）",
                                 H1, T1, y1, yt1, nodes1, cls1, None, 100, [32, 64, 96])

    with open(os.path.join(ROOT, "results", "results_metric_learn2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n已保存 results_metric_learn2.json")


if __name__ == "__main__":
    main()
