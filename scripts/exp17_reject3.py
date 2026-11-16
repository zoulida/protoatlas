"""exp17: 拒识还有没有救？
三个候选信号，都在同一批 ID / OOD 上测：
  A 到本地最近训练样本的距离（kNN 密度，而非到原型距离）
  B top-3 候选节点的投票一致性（分布式判据，单节点做不到）
  C 单节点 head margin（上一轮已知最弱，此处作基线）
"""
import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def auroc(si, so):
    s = np.concatenate([si, so])
    y = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    r = np.empty(len(s), np.float64)
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_at_frr(si, so, frr=0.05):
    """在 ID 只看错 frr 的阈值下，OOD 被挡住多少。score 越高越像 ID。"""
    thr = np.percentile(si, 100 * frr)
    return float(np.mean(so < thr))


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    Q = np.load(os.path.join(CACHE, "cifar100.npz"))["x"].astype(np.float32) if False else None

    O_c100 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:2000]
    O_mnist = np.load(os.path.join(CACHE, "feat_mnist.npy")).astype(np.float32)[:2000]

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P, pcls, pnode = np.stack(protos).astype(np.float32), np.array(pcls), np.array(pnode)

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    Zp = (P - mu[0]) @ W

    # 每个节点的本地训练特征（原始 384 维，算 kNN 密度）
    local = {n: H[tr_node == n] for n in range(C["n_nodes"])}
    heads = {n: train_head(H[tr_node == n],
                           np.array([sorted(nc[n]).index(c) for c in y_tr[tr_node == n]]))
             for n in range(C["n_nodes"])}
    node_of_cls = {}
    for j, c in enumerate(pcls):
        node_of_cls.setdefault(c, []).append(pnode[j])

    def knn_min_dist(X, n, chunk=256):
        """X 中每个样本到节点 n 本地训练特征的最近距离（384 维原始空间）"""
        L = local[n]
        out = np.empty(len(X), np.float32)
        ln = (L ** 2).sum(1)
        for i in range(0, len(X), chunk):
            a = (X[i:i + chunk] ** 2).sum(1)[:, None] + ln[None, :] - 2.0 * (X[i:i + chunk] @ L.T)
            out[i:i + chunk] = np.sqrt(np.maximum(a.min(1), 0))
        return out

    def pred(n, X):
        with torch.no_grad():
            return heads[n](torch.from_numpy(X)).numpy().argmax(1)

    def route_cands(z, k=3):
        d = dist(z[None, :], Zp)[0]
        seen, out = set(), []
        for j in np.argsort(d):
            c = pcls[j]
            if c not in seen:
                seen.add(c)
                out.append((c, pnode[j], d[j]))
            if len(out) == k:
                break
        return out

    log(f"{'信号':<34}{'AUROC':>9}{'5%误拒下拦截':>14}")
    log("-" * 60)
    res = {}
    for name, O in [("cifar100", O_c100), ("mnist", O_mnist)]:
        log(f"\n=== OOD = {name} ===")
        best = {}
        for label, X in [("ID", T), ("OOD", O)]:
            Zo = (X - mu[0]) @ W
            # A: 到 top1 节点本地样本的最近距离（取负，越大越像 ID）
            sA, sB, sC = [], [], []
            for i in range(len(X)):
                c0, n0, _ = route_cands(Zo[i])[0]
                sA.append(-knn_min_dist(X[i:i + 1], n0)[0])
                # B: 同一个类由持有它的多个节点**独立**判断，看答案一致率
                #    （去重后的 3 个候选是不同类，没法投票；必须换成"同类多节点"）
                votes = []
                for n in node_of_cls.get(c0, [n0]):
                    cls = sorted(nc[n])
                    votes.append(cls[pred(n, X[i:i + 1])[0]])
                cnt = max(np.bincount(np.array(votes))) if votes else 0
                sB.append(cnt / max(len(votes), 1))
                # C: 单节点 margin
                with torch.no_grad():
                    lg = heads[n0](torch.from_numpy(X[i:i + 1])).numpy()[0]
                sC.append(lg.max() - np.partition(lg, -2)[-2])
            best[label] = (np.array(sA), np.array(sB, float), np.array(sC))
        for k, nm in [(0, "A kNN 距离（到最近本地样本）"),
                      (1, "B top-3 投票全一致"),
                      (2, "C 单节点 margin（基线）")]:
            si, so = best["ID"][k], best["OOD"][k]
            a = auroc(si, so)
            b = block_at_frr(si, so)
            log(f"  {nm:<32}{a:>9.3f}{b:>14.3f}")
            res[f"{name}_{nm[0]}"] = dict(auroc=a, block_at_5frr=b)

    with open(os.path.join(ROOT, "results", "results_reject3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
