"""exp22: 类别变多会怎样？—— 100 类扩展 + 双层路由是否值得

对照设定（节点数相同，只增加每节点类数）：
  CIFAR-10 : 20 节点 × 2 类（已知：top-1 0.954 / top-3 去重 0.994）
  CIFAR-100: 20 节点 × 5 类（shard-5）
  CIFAR-100: 10 节点 × 10 类（shard-10，极端）

三个问题：
  ① 类别变多时，top-k 候选召回退化多快？k 要多大才够？
  ② 拒识（energy）在类多时还是否有效？
  ③ 双层路由（20 超类 → 类内细类）能否把退化补回来？
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


def auroc(si, so):
    s = np.concatenate([si, so])
    y = np.concatenate([np.ones(len(si)), np.zeros(len(so))])
    r = np.empty(len(s), np.float64)
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dedup_recall(order, pcls, ytrue, k):
    """按类去重取前 k 个类，正确类是否在里面"""
    hit = 0
    for i in range(len(ytrue)):
        seen = []
        for j in order[i]:
            c = pcls[j]
            if c not in seen:
                seen.append(c)
                if c == ytrue[i]:
                    hit += 1
                    break
            if len(seen) == k:
                break
    return hit / len(ytrue)


def main():
    import pandas as pd

    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr, ctr = d["y"].astype(np.int32), d["coarse"].astype(np.int32)
    yte = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)
    cte = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["coarse_label"] \
        .to_numpy().astype(np.int32)
    n_te = min(len(Hte), len(yte))
    Hte, yte, cte = Hte[:n_te], yte[:n_te], cte[:n_te]
    log(f"训练 {Htr.shape}  测试 {Hte.shape}  细类 {len(np.unique(ytr))}  超类 {len(np.unique(ctr))}")

    # 坐标系：用 CIFAR-100 训练集拟合（等价于"公开参照集"）
    mu = Htr.mean(0, keepdims=True)
    Cc = (Htr - mu).T @ (Htr - mu) / (len(Htr) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Ztr, Zte = (Htr - mu[0]) @ W, (Hte - mu[0]) @ W

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    res = {}
    # ---------- 自适应 K 的候选召回 ----------
    for shard, nnode in [(5, 20), (10, 10), (2, 50)]:
        # 按类顺序切片分配：节点 n 拿第 n*shard ~ (n+1)*shard 类
        nodes = [list(range(n * shard, min((n + 1) * shard, 100))) for n in range(nnode)]
        protos, pcls, pnode = [], [], []
        for n, cs in enumerate(nodes):
            for c in cs:
                m = ytr == c
                if m.sum() == 0:
                    continue
                protos.append(Htr[m].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls_np, pnode_np = np.array(pcls), np.array(pnode)
        Zp = (P - mu[0]) @ W
        D = dist(Zte, Zp)
        order = np.argsort(D, 1)
        top1 = float(np.mean(pcls_np[order[:, 0]] == yte))

        log("\n" + "=" * 70)
        log(f"=== shard-{shard}：{nnode} 节点 × 每节点 {shard} 类（共 100 类）===")
        log(f"  top-1 命中 {top1:.4f}")
        row = dict(top1=top1, nnode=nnode, shard=shard)
        for k in [1, 2, 3, 5, 10, 20]:
            r = dedup_recall(order, pcls_np, yte, k)
            row[f"recall@{k}"] = r
            log(f"  候选召回@{k:<2} {r:.4f}")
        res[f"shard{shard}"] = row

        # energy 拒识：正类 = 路由对，负类 = CIFAR-10 测试集（跨数据集 OOD）
        # 用"到最近原型的距离"和"energy"两种；此处只测 energy（需 head，跳过）
        # → 用简化代理：到 top-1 原型的距离
        dmin = D[np.arange(len(Zte)), order[:, 0]]
        is_id = pcls_np[order[:, 0]] == yte
        res[f"shard{shard}"]["geo_auroc"] = auroc(-dmin[is_id], -dmin[~is_id])
        log(f"  （诊断）到 top-1 原型距离的 AUROC {res[f'shard{shard}']['geo_auroc']:.3f}")

    # ---------- 双层路由（20 超类 → 类内） ----------
    log("\n" + "=" * 70)
    log("=== 双层路由：先 20 超类，再类内细类 ===")
    cprotos = np.stack([Htr[ctr == c].mean(0) for c in range(20)]).astype(np.float32)
    Zc = (cprotos - mu[0]) @ W
    Dc = dist(Zte, Zc)
    corder = np.argsort(Dc, 1)

    for n_c in [1, 2, 3, 5]:
        # 先取 n_c 个超类，再在这些超类包含的细类里找 top-3
        fine_ids = [np.where(np.isin(ctr, corder[i, :n_c]))[0] for i in range(len(Zte))]
        hit, asked = 0, 0
        for i in range(len(Zte)):
            fids = fine_ids[i]
            if len(fids) == 0:
                continue
            cand = [c for c in np.unique(ytr[fids])]
            pts = np.stack([Htr[fids][ytr[fids] == c].mean(0) for c in cand]).astype(np.float32)
            dd = dist(Zte[i:i + 1], (pts - mu[0]) @ W)[0]
            o = np.argsort(dd)
            seen = []
            for j in o:
                cc = cand[j]
                if cc not in seen:
                    seen.append(cc)
                if len(seen) == 3:
                    break
            if yte[i] in seen:
                hit += 1
            asked += len(o)
        log(f"  超类候选 {n_c:>2} → 细类 top-3 召回 {hit/len(Zte):.4f}"
            f"（平均比较 {asked/len(Zte):.0f} 个细类原型）")
        res[f"hier_nc{n_c}"] = dict(recall=hit / len(Zte), compared=asked / len(Zte))

    with open(os.path.join(ROOT, "results", "results_c100_scale.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
