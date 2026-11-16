"""exp22b: 快速对照 —— 类别数 10 → 100，候选召回退化多快？

统一"每类 15 张算原型、其余查询"，使各组唯一变量是类别结构：
  A  CIFAR-10   : 5 节点 × 2 类   = 10 原型
  B  CIFAR-100  : 50 节点 × 2 类  = 100 原型
  C  CIFAR-100  : 20 节点 × 5 类  = 100 原型
  D  CIFAR-100  : 10 节点 × 10 类 = 100 原型

⚠️ 每类 15 张原型比正式实验（300~500 张）噪声大，绝对精度会偏低，
   但四组同口径，趋势可比。正式版用训练集特征重跑。

另测双层路由（官方 20 超类 → 类内细类）。
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


def fit_W(H, k=16):
    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    return mu[0], V[:, -k:].astype(np.float32)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def first_rank(order, pcls, ytrue, kmax=20):
    first = np.full(len(ytrue), 10 ** 6, np.int32)
    for i in range(len(ytrue)):
        seen, cnt = set(), 0
        for j in order[i]:
            c = pcls[j]
            if c in seen:
                continue
            seen.add(c)
            cnt += 1
            if c == ytrue[i]:
                first[i] = cnt
                break
            if cnt >= kmax:
                break
    return first


def main():
    import pandas as pd
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    T10 = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y10 = yte[te_sel].astype(np.int32)
    assert len(T10) == len(y10), (T10.shape, y10.shape)

    T100 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    df = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    y100 = df["fine_label"].to_numpy().astype(np.int32)[:3000]
    c100 = df["coarse_label"].to_numpy().astype(np.int32)[:3000]
    log(f"CIFAR-10 {T10.shape} 类 {len(np.unique(y10))} | "
        f"CIFAR-100 {T100.shape} 类 {len(np.unique(y100))} 超类 {len(np.unique(c100))}")
    log(f"fine//5 vs coarse 一致率 {np.mean((y100//5)==c100):.3f}（低说明细类不按超类成块，必须用官方 coarse）")

    res = {}

    def run(tag, T, y, n_cls, shard, nnode, nproto=15):
        rng = np.random.RandomState(0)
        pidx, qidx = [], []
        for c in range(n_cls):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            pidx += list(idx[:nproto])
            qidx += list(idx[nproto:])
        pidx, qidx = np.array(pidx), np.array(qidx)
        Hp, yp = T[pidx], y[pidx]
        mu, W = fit_W(T, 16)
        Zq, yq = (T[qidx] - mu) @ W, y[qidx]

        nodes = [list(range(n * shard, min((n + 1) * shard, n_cls))) for n in range(nnode)]
        protos, pcls, pnode = [], [], []
        for n, cs in enumerate(nodes):
            for c in cs:
                m = yp == c
                if m.sum() > 0:
                    protos.append(Hp[m].mean(0))
                    pcls.append(c)
                    pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls = np.array(pcls)
        order = np.argsort(dist(Zq, (P - mu) @ W), 1)
        top1 = float(np.mean(pcls[order[:, 0]] == yq))
        first = first_rank(order, pcls, yq)

        log("\n" + "=" * 72)
        log(f"=== {tag} ===  {len(protos)} 原型 / {len(yq)} 查询")
        s = f"  top-1 {top1:.4f}   "
        rr = {}
        for k in [1, 2, 3, 5, 10, 20]:
            v = float(np.mean(first <= k))
            rr[k] = v
            s += f"@{k} {v:.4f}  "
        log(s)
        res[tag] = dict(top1=top1, n_proto=len(protos), n_query=int(len(yq)),
                        **{f"r{k}": v for k, v in rr.items()})
        return dict(mu=mu, W=W, Hp=Hp, yp=yp, pcls=pcls, cls_ids=sorted(np.unique(yp)))

    run("A CIFAR-10   5节点×2类", T10, y10, 10, 2, 5)
    st = run("B CIFAR-100 50节点×2类", T100, y100, 100, 2, 50)
    run("C CIFAR-100 20节点×5类", T100, y100, 100, 5, 20)
    run("D CIFAR-100 10节点×10类", T100, y100, 100, 10, 10)

    # ---- 双层 ----
    log("\n" + "=" * 72)
    log("=== 双层路由（CIFAR-100，官方 20 超类 → 类内细类）===")
    mu, W, Hp, yp = st["mu"], st["W"], st["Hp"], st["yp"]
    rng = np.random.RandomState(0)
    pidx, qidx = [], []
    for c in range(100):
        idx = np.where(y100 == c)[0]
        rng.shuffle(idx)
        pidx += list(idx[:15])
        qidx += list(idx[15:])
    pidx, qidx = np.array(pidx), np.array(qidx)
    cp = c100[pidx]
    cls_ids = sorted(np.unique(yp))
    Pcls = np.stack([Hp[yp == c].mean(0) for c in cls_ids]).astype(np.float32)
    cls_coarse = np.array([int(cp[yp == c][0]) for c in cls_ids])
    Zp = (Pcls - mu) @ W
    co_ids = sorted(np.unique(cp))
    CP = np.stack([Hp[cp == c].mean(0) for c in co_ids]).astype(np.float32)
    Zq, yq = (T100[qidx] - mu) @ W, y100[qidx]
    co_order = np.argsort(dist(Zq, (CP - mu) @ W), 1)

    for nk in [1, 2, 3, 5]:
        hits, cmp_cnt = 0, 0
        for i in range(len(Zq)):
            keep = np.zeros(len(cls_ids), bool)
            for j in co_order[i, :nk]:
                keep |= (cls_coarse == co_ids[j])
            sub = np.where(keep)[0]
            if len(sub) == 0:
                continue
            cmp_cnt += len(sub)
            o = [sub[j] for j in np.argsort(dist(Zq[i:i + 1], Zp[sub])[0])]
            seen = []
            for k in o:
                if cls_ids[k] not in seen:
                    seen.append(cls_ids[k])
                if len(seen) == 3:
                    break
            if yq[i] in seen:
                hits += 1
        log(f"  超类@{nk} → 细类top3 召回 {hits/len(Zq):.4f}"
            f"   平均比 {cmp_cnt/len(Zq):.1f} 个细类原型（单级要比 {len(cls_ids)} 个）")
        res[f"hier_nc{nk}"] = dict(recall=hits / len(Zq), compared=cmp_cnt / len(Zq))

    with open(os.path.join(ROOT, "results", "results_c100_quick.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
