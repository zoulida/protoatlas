"""exp19: 把"本地误识"拆开，并测拒识之后怎么处理

① 误识解剖
   路由把样本送到节点 n1（最近原型属主）。它的真实类 y 与 n1 的关系有三种：
     A  c1 == y                    → 路由对
     B  c1 != y 且 y ∈ 本节点的类   → 送错了类，但 n1 也认识 y（可能答对）
     C  c1 != y 且 y ∉ 本节点的类   → 知识缺失，n1 必然答错
   只有 C 是真的"本地不认识"。

② 拒识后的处理策略
   hard / reject_only / fallback3 / fallback10 / vote3 / soft_top3
   指标：覆盖率、错误率、明确"不知道"率、平均询问节点数
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
    NAME = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
    N = len(y_te)

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
    Zp, Zq = (P - mu[0]) @ W, (T - mu[0]) @ W

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    D = dist(Zq, Zp)
    order = np.argsort(D, axis=1)

    # 每节点 head 对所有测试样本的预测（全局类标）
    log("预计算 20 个节点的 head 预测矩阵...")
    pred = np.zeros((C["n_nodes"], N), np.int32)
    for n in range(C["n_nodes"]):
        m = tr_node == n
        cls = sorted(nc[n])
        head = train_head(H[m], np.array([cls.index(c) for c in y_tr[m]]))
        out = []
        with torch.no_grad():
            for i in range(0, N, 512):
                lg = head(torch.from_numpy(T[i:i + 512])).numpy().argmax(1)
                out.append(np.array(cls)[lg])
        pred[n] = np.concatenate(out)

    # 每节点的 kNN 距离与阈值（放行 95% 自家样本）
    kd = np.zeros((C["n_nodes"], N), np.float32)
    for n in range(C["n_nodes"]):
        L = H[tr_node == n]
        ln = (L ** 2).sum(1)
        for i in range(0, N, 512):
            a = (T[i:i + 512] ** 2).sum(1)[:, None] + ln[None, :] - 2.0 * (T[i:i + 512] @ L.T)
            kd[n, i:i + 512] = np.sqrt(np.maximum(a.min(1), 0))
    id_idx = {n: [i for i in range(N) if y_te[i] in cls_of_node[n]] for n in range(C["n_nodes"])}
    thr = {n: float(np.percentile(kd[n][id_idx[n]], 95)) for n in range(C["n_nodes"])}

    n1 = pnode[order[:, 0]]
    c1 = pcls[order[:, 0]]
    has_y = np.array([y_te[i] in cls_of_node[n1[i]] for i in range(N)])

    A = c1 == y_te
    B = (~A) & has_y
    Cq = (~A) & (~has_y)

    log("\n" + "=" * 76)
    log("① 本地误识解剖（402.4 万次判断的四象限）")
    log("=" * 76)
    log(f"  A 路由对（c1 == 真实类）          {A.mean():.4f}  ({A.sum():5d})")
    log(f"  B 送错类，但该节点也持有真实类      {B.mean():.4f}  ({B.sum():5d})  ← 良性")
    log(f"  C 送错类，且该节点不认识真实类      {Cq.mean():.4f}  ({Cq.sum():5d})  ← 真危险")
    log(f"\n  合计路由错 {A.mean() and (1-A.mean()):.4f}，其中 "
        f"{Cq.sum()/max((~A).sum(),1)*100:.1f}% 是知识缺失，"
        f"{B.sum()/max((~A).sum(),1)*100:.1f}% 是良性")

    # B 象限里 head 实际答对多少（送错类但仍可能答对）
    b_ok = float(np.mean(pred[n1[B], np.where(B)[0]] == y_te[B])) if B.sum() else 0.0
    log(f"  B 象限里 head 实际答对 {b_ok:.4f}（真实类在本地，所以多数能答对）")
    log(f"  C 象限里 head 答对 {float(np.mean(pred[n1[Cq], np.where(Cq)[0]] == y_te[Cq])):.4f}"
        f"（结构上必然为 0：它只能输出自己那 2 类）")

    # 节点是否拒识，与象限的关系
    log("\n  拒识（kNN 95% 阈值）在各象限的表现：")
    acc_mask = np.array([kd[n1[i], i] <= thr[n1[i]] for i in range(N)])
    for nm, mask in [("A 路由对", A), ("B 良性", B), ("C 知识缺失", Cq)]:
        if mask.sum() == 0:
            continue
        log(f"    {nm:<14} 受理率 {acc_mask[mask].mean():.4f}")

    # ------------------ ② 拒识后的处理策略 ------------------
    def cands(i, k=3):
        seen, out = set(), []
        for j in order[i]:
            c = pcls[j]
            if c not in seen:
                seen.add(c)
                out.append((c, pnode[j]))
            if len(out) == k:
                break
        return out

    log("\n" + "=" * 76)
    log("② 拒识之后怎么处理（kNN 判据，放行 95%）")
    log("=" * 76)
    log(f"  {'策略':<16}{'覆盖':>8}{'错误':>8}{'不知道':>8}{'问节点数':>9}")

    res = {}

    def report(name, right, wrong, unknown, asked):
        tot = len(right)
        log(f"  {name:<16}{np.mean(right):>8.4f}{np.mean(wrong):>8.4f}"
            f"{np.mean(unknown):>8.4f}{np.mean(asked):>9.2f}")
        res[name] = dict(coverage=float(np.mean(right)), error=float(np.mean(wrong)),
                         unknown=float(np.mean(unknown)), asked=float(np.mean(asked)))

    # 1 hard：问 top-1，直接答
    for label, k, stop_on_accept in [("hard（问1个硬答）", 1, True),
                                     ("reject_only（拒就弃）", 1, True),
                                     ("fallback3（依次问3个）", 3, True),
                                     ("fallback10（问10个类）", 10, True),
                                     ("vote3（3个投票）", 3, False)]:
        right, wrong, unknown, asked = [], [], [], []
        for i in range(N):
            cs = cands(i, k)
            if label.startswith("hard"):
                a = 1
                right.append(int(pred[cs[0][1], i] == y_te[i]))
                wrong.append(int(pred[cs[0][1], i] != y_te[i]))
                unknown.append(0)
                asked.append(1)
                continue
            if label.startswith("vote3"):
                votes = [pred[n, i] for _, n in cs]
                cnt = np.bincount(np.array(votes), minlength=10)
                top = int(np.argmax(cnt))
                right.append(int(top == y_te[i]))
                wrong.append(int(top != y_te[i] and cnt.max() > len(votes) / 2))
                unknown.append(int(cnt.max() <= len(votes) / 2))
                asked.append(len(cs))
                continue
            got, q = None, 0
            for c, n in cs:
                q += 1
                if kd[n, i] <= thr[n]:
                    got = pred[n, i]
                    break
            right.append(int(got is not None and got == y_te[i]))
            wrong.append(int(got is not None and got != y_te[i]))
            unknown.append(int(got is None))
            asked.append(q)
        report(label, right, wrong, unknown, asked)

    # soft：不硬判，返回 top-3 类，看正确类在不在里面
    hit = [int(y_te[i] in [c for c, _ in cands(i, 3)]) for i in range(N)]
    log(f"  {'soft_top3（给3候选）':<16}{np.mean(hit):>8.4f}{0.0:>8.4f}{1-np.mean(hit):>8.4f}"
        f"{3.0:>9.2f}   ← 不硬判")
    res["soft_top3"] = dict(coverage=float(np.mean(hit)), error=0.0,
                            unknown=float(1 - np.mean(hit)), asked=3.0)

    with open(os.path.join(ROOT, "results", "results_disambig.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
