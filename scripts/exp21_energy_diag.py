"""exp21: 为什么 energy 会反向？—— 机制诊断

假设：每节点 head 只在 2 类上训练，是**无界线性层**。
OOD 样本落在训练分布之外，特征范数更大 → |logit| 爆炸 → logsumexp 大 →
energy = -logsumexp 很低 → 被判成"极度自信的 ID"。

三个量一起看：||h||、max|logit|、logsumexp。
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    Nn = C["n_nodes"]
    O_c100 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:2000]
    O_mnist = np.load(os.path.join(CACHE, "feat_mnist.npy")).astype(np.float32)[:2000]

    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zp = (P - mu[0]) @ W

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_head(H[m], np.array([sorted(nc[n]).index(c) for c in y_tr[m]]))
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    def diag(X, tag):
        n1 = pnode[np.argmin(dist((X - mu[0]) @ W, Zp), 1)]
        norms = np.linalg.norm(X, axis=1)
        lmax = np.zeros(len(X), np.float32)
        lse = np.zeros(len(X), np.float32)
        with torch.no_grad():
            for n in np.unique(n1):
                idx = np.where(n1 == n)[0]
                lg = heads[n](torch.from_numpy(X[idx])).numpy()
                lmax[idx] = np.abs(lg).max(1)
                m_ = lg.max(1, keepdims=True)
                lse[idx] = (m_[:, 0] + np.log(np.exp(lg - m_).sum(1)))
        log(f"  {tag:<12}{norms.mean():>9.2f}{np.median(norms):>9.2f}"
            f"{lmax.mean():>10.2f}{lse.mean():>10.2f}{(-lse).mean():>11.2f}")
        return dict(norm=float(norms.mean()), lmax=float(lmax.mean()), lse=float(lse.mean()))

    # ID 只取"路由对"的样本
    n1_T = pnode[np.argmin(dist((T - mu[0]) @ W, Zp), 1)]
    c1_T = pcls[np.argmin(dist((T - mu[0]) @ W, Zp), 1)]
    A = c1_T == y_te

    log(f"  {'集合':<12}{'‖h‖均值':>9}{'‖h‖中位':>9}{'max|logit|':>10}{'logsumexp':>10}{'energy':>11}")
    diag(T[A], "ID（路由对）")
    diag(T[~A], "路由错投")
    diag(O_c100, "CIFAR-100")
    diag(O_mnist, "MNIST")

    # head 权重的尺度
    wn = [float(np.linalg.norm(heads[n].weight.detach().numpy(), axis=1).mean()) for n in range(Nn)]
    log(f"\n  head 权重行范数均值 {np.mean(wn):.3f}（20 个节点，范围 {min(wn):.2f}~{max(wn):.2f}）")
    log("  → 若 OOD 的 ‖h‖ 更大且 |logit| 随之增大，即证实'线性层无界 → 过度自信'")


if __name__ == "__main__":
    main()
