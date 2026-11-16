"""exp20: energy 能否做拒识判据？与 MSP / margin / kNN / 组合正面比较

核心问题：energy = -logsumexp(logits) 是"跨节点不可比的绝对量"。
每个节点 head 的训练集不同、logits 尺度不同，直接合并统计会得出错误结论。
故所有判据都做 per-node 标定：把分数映射成"在该节点自家类 ID 样本上的分位"，
再合并算 AUROC。

三组对照：
  ① 近域 OOD（CIFAR-100，100 类，与 CIFAR-10 同为自然图像）
  ② 远域 OOD（MNIST，灰度手写数字）
  ③ 路由错投（C 象限：节点不持有真实类）—— 这才是"危险错答"的来源
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
    """score 越高越像 ID"""
    s = np.concatenate([si, so])
    y = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    r = np.empty(len(s), np.float64)
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_at_frr(si, so, frr=0.05):
    thr = np.percentile(si, 100 * frr)
    return float(np.mean(so < thr))


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

    # ---- 原型 / 坐标 ----
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

    # ---- heads ----
    heads, local = {}, {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_head(H[m], np.array([sorted(nc[n]).index(c) for c in y_tr[m]]))
        local[n] = H[m]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    def route_top1(Z):
        D = dist(Z, Zp)
        o = np.argmin(D, 1)
        return pnode[o], pcls[o]

    def signals(X, n_arr):
        """对每个样本，只在它被路由到的那个节点上算 6 个信号（原始空间 384 维）"""
        n_s = len(X)
        msp = np.zeros(n_s, np.float32)
        mgn = np.zeros(n_s, np.float32)
        eng = np.zeros(n_s, np.float32)
        knn = np.zeros(n_s, np.float32)
        for n in np.unique(n_arr):
            idx = np.where(n_arr == n)[0]
            Xi = X[idx]
            with torch.no_grad():
                lg = heads[n](torch.from_numpy(Xi)).numpy()
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            msp[idx] = p.max(1)
            srt = np.sort(lg, 1)
            mgn[idx] = srt[:, -1] - srt[:, -2]
            # energy = -logsumexp(logits)；等价地 logsumexp 越大越像 ID
            lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
            eng[idx] = lse          # 直接用 logsumexp：越高越像 ID
            L = local[n]
            ln = (L ** 2).sum(1)
            a = (Xi ** 2).sum(1)[:, None] + ln[None, :] - 2.0 * (Xi @ L.T)
            knn[idx] = np.sqrt(np.maximum(a.min(1), 0))
        return msp, mgn, eng, -knn   # 统一成"越高越像 ID"

    # ---- 参考分布：每节点自家类 ID 样本（用于 per-node 标定）----
    n1_T, c1_T = route_top1((T - mu[0]) @ W)
    ref_msp, ref_mgn, ref_eng, ref_knn = signals(T, n1_T)
    has_y_T = np.array([y_te[i] in cls_of_node[n1_T[i]] for i in range(len(T))])
    # 参考集只用"路由对"的样本，避免把错投样本混进标定
    A_mask = c1_T == y_te

    # 按节点收集参考分位表
    def make_ecdf(sig, n_arr):
        tab = {}
        for n in range(Nn):
            v = sig[n_arr == n]
            tab[n] = np.sort(v) if len(v) >= 20 else None
        return tab

    def to_pct(sig, tab, n_arr):
        out = np.full(len(sig), 0.5, np.float32)
        for n in range(Nn):
            t = tab[n]
            if t is None:
                continue
            idx = np.where(n_arr == n)[0]
            if len(idx) == 0:
                continue
            out[idx] = np.searchsorted(t, sig[idx]) / len(t)
        return out

    log(f"参考集：路由对样本 {A_mask.sum()} / {len(T)}；每节点平均 {A_mask.sum()/Nn:.0f} 张")

    # ---- 三组对照 ----
    groups = {}
    # ① 近域 / ② 远域：ID = A 象限（真正自己的东西）
    id_mask = A_mask
    for name, O in [("CIFAR-100 近域", O_c100), ("MNIST 远域", O_mnist)]:
        n1_O, _ = route_top1((O - mu[0]) @ W)
        s_id = signals(T[id_mask], n1_T[id_mask])
        s_od = signals(O, n1_O)
        groups[name] = ("ood", s_id, n1_T[id_mask], s_od, n1_O)
    # ③ 路由错投：正类 = A 象限，负类 = C 象限（不持有真实类）
    Cq = (~A_mask) & (~has_y_T)
    groups["路由错投 C 象限"] = ("route", signals(T[id_mask], n1_T[id_mask]),
                                n1_T[id_mask], signals(T[Cq], n1_T[Cq]), n1_T[Cq])

    NAMES = ["MSP", "margin", "energy", "kNN 距离", "energy+kNN"]
    res = {}
    for gname, (kind, s_id, n_id, s_neg, n_neg) in groups.items():
        # per-node 标定
        tabs = [make_ecdf(s_id[k], n_id) for k in range(4)]
        P_id = [to_pct(s_id[k], tabs[k], n_id) for k in range(4)]
        P_neg = [to_pct(s_neg[k], tabs[k], n_neg) for k in range(4)]
        # 组合：energy 与 kNN 等权
        P_id.append(0.5 * P_id[2] + 0.5 * P_id[3])
        P_neg.append(0.5 * P_neg[2] + 0.5 * P_neg[3])

        log("\n" + "=" * 66)
        log(f"=== {gname} ===  (正类 {len(P_id[0])}，负类 {len(P_neg[0])})")
        log(f"  {'判据':<14}{'AUROC':>9}{'5%误拒拦截':>13}")
        for k, nm in enumerate(NAMES):
            a = auroc(P_id[k], P_neg[k])
            b = block_at_frr(P_id[k], P_neg[k])
            log(f"  {nm:<14}{a:>9.3f}{b:>13.3f}")
            res[f"{gname}|{nm}"] = dict(auroc=a, block_at_5frr=b)

    # ---- 不做 per-node 标定的 energy（用来证明"不标定会错"）----
    n1_O, _ = route_top1((O_c100 - mu[0]) @ W)
    raw_id = signals(T[id_mask], n1_T[id_mask])[2]
    raw_od = signals(O_c100, n1_O)[2]
    log("\n" + "-" * 66)
    log(f"【对照】energy 不做 per-node 标定，直接合并：AUROC {auroc(raw_id, raw_od):.3f}")
    log(f"       做 per-node 标定后：AUROC {res['CIFAR-100 近域|energy']['auroc']:.3f}")
    res["未标定_energy_auroc"] = auroc(raw_id, raw_od)

    with open(os.path.join(ROOT, "results", "results_energy.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
