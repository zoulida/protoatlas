"""exp18: 端到端最终表 —— 拒识判据选对之后，路由错那 5.7% 能救回多少

判据三种：
  none   不拒识，谁被问到谁答
  proto  到本节点该类**原型**的距离 <= 阈值（上一轮用的，失效）
  knn    到本节点**最近训练样本**的距离 <= 阈值（本轮修正）
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


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P, pcls, pnode = np.stack(protos).astype(np.float32), np.array(pcls), np.array(pnode)
    cls_of_node = {n: sorted(nc[n]) for n in range(C["n_nodes"])}

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    Zp = (P - mu[0]) @ W
    Zq = (T - mu[0]) @ W
    D = dist(Zq, Zp)
    order = np.argsort(D, axis=1)

    # 预计算：每个节点对所有测试样本的
    #   kd[n][i]  = 到节点 n 最近训练样本的距离（384 维原始特征）
    #   pd_[n,j]  = 到"节点 n 的类 j 原型"的距离（沿用它做对照）
    log("预计算每节点的 kNN 距离与原型距离...")
    kd = np.zeros((C["n_nodes"], len(T)), np.float32)
    for n in range(C["n_nodes"]):
        L = H[tr_node == n]
        ln = (L ** 2).sum(1)
        dm = (T ** 2).sum(1)[:, None] + ln[None, :] - 2.0 * (T @ L.T)
        kd[n] = np.sqrt(np.maximum(dm.min(1), 0))

    # 原型距离（在 16 维坐标上，与路由同一空间）
    pj = {}
    for j in range(len(P)):
        pj[(pnode[j], pcls[j])] = j

    def proto_d(i, n, c):
        return float(np.sqrt(((Zq[i] - Zp[pj[(n, c)]]) ** 2).sum()))

    # 阈值：只用"属于该节点类"的样本标定（放行 95% 的自家样本）
    # ⚠️ 若用全部测试样本标定，其中 4/5 是别人的东西，阈值会被抬到几乎不拒
    id_idx = {n: [i for i in range(len(T)) if y_te[i] in cls_of_node[n]]
              for n in range(C["n_nodes"])}
    thr_k = {n: float(np.percentile(kd[n][id_idx[n]], 95)) for n in range(C["n_nodes"])}
    thr_p = {}
    for n in range(C["n_nodes"]):
        vals = [min(proto_d(i, n, c) for c in cls_of_node[n]) for i in id_idx[n]]
        thr_p[n] = float(np.percentile(vals, 95))

    # 对照：只用同一个 ID 集合，看 kNN 距离能把"别人类的请求"分出来多少
    own, oth = [], []
    for n in range(C["n_nodes"]):
        own.append(kd[n][id_idx[n]])
        rest = [i for i in range(len(T)) if y_te[i] not in cls_of_node[n]]
        oth.append(kd[n][rest])
    own, oth = np.concatenate(own), np.concatenate(oth)
    log(f"\n同一节点上：自家类样本的距离中位 {np.median(own):.2f}，"
        f"别人类样本 {np.median(oth):.2f}（比值 {np.median(oth)/np.median(own):.2f}）")
    log(f"放行 95% 自家样本时，别人类样本被挡 {np.mean(oth > np.percentile(own, 95)):.3f}")

    log("\n" + "=" * 74)
    log("端到端：按类去重 top-3 依次询问，第一个受理的答（PCA-16 路由）")
    log("=" * 74)
    log(f"{'判据':<10}{'答对':>9}{'危险错误':>11}{'全拒':>9}{'└ 错送被挡':>12}")
    log("-" * 74)
    res = {}
    for mode in ["none", "proto", "knn"]:
        ok = wrong = rej = blocked = 0
        for i in range(len(y_te)):
            seen, cand = set(), []
            for j in order[i]:
                c = pcls[j]
                if c not in seen:
                    seen.add(c)
                    cand.append((c, pnode[j]))
                if len(cand) == 3:
                    break
            got = None
            for pos, (c, n) in enumerate(cand):
                if mode == "none":
                    accept = True
                elif mode == "proto":
                    accept = proto_d(i, n, c) <= thr_p[n]
                else:
                    accept = kd[n, i] <= thr_k[n]
                if pos == 0 and not accept and c != y_te[i]:
                    blocked += 1
                if accept:
                    got = c
                    break
            if got is None:
                rej += 1
            elif got == y_te[i]:
                ok += 1
            else:
                wrong += 1
        N = len(y_te)
        res[mode] = dict(correct=ok / N, dangerous=wrong / N, all_rejected=rej / N,
                         wrong_blocked=blocked / N)
        log(f"{mode:<10}{ok/N:>9.4f}{wrong/N:>11.4f}{rej/N:>9.4f}{blocked/N:>12.4f}")

    # 阈值松紧对 knn 判据的影响
    log("\n" + "=" * 74)
    log("kNN 判据的放行比例扫描")
    log("=" * 74)
    log(f"{'放行%':<9}{'答对':>9}{'危险错误':>11}{'全拒':>9}")
    for pr in [80, 90, 95, 99]:
        thr = {n: float(np.percentile(kd[n][id_idx[n]], pr)) for n in range(C["n_nodes"])}
        ok = wrong = rej = 0
        for i in range(len(y_te)):
            seen, cand = set(), []
            for j in order[i]:
                c = pcls[j]
                if c not in seen:
                    seen.add(c)
                    cand.append((c, pnode[j]))
                if len(cand) == 3:
                    break
            got = None
            for c, n in cand:
                if kd[n, i] <= thr[n]:
                    got = c
                    break
            if got is None:
                rej += 1
            elif got == y_te[i]:
                ok += 1
            else:
                wrong += 1
        N = len(y_te)
        log(f"{pr:<9}{ok/N:>9.4f}{wrong/N:>11.4f}{rej/N:>9.4f}")
        res[f"knn_pass{pr}"] = dict(correct=ok / N, dangerous=wrong / N,
                                    all_rejected=rej / N)

    with open(os.path.join(ROOT, "results", "results_end2end.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
