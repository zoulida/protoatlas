"""exp23: 正式版 —— 每类 500 张原型，10 类 vs 100 类完全同口径

  CIFAR-10 : 训练 12000（每类 1200），取 500/类 算原型；查询 5000
  CIFAR-100: 训练 50000（每类 500），全用算原型；查询 3000

回答三件事：
  ① 原型质量正常时，100 类的召回@k 到底多少（快照版用 15 张，偏低）
  ② 10 类 vs 100 类的真实差距
  ③ 双层路由（官方 20 超类）在正常原型下是否仍为负收益
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


def first_rank(order, pcls, ytrue, kmax=30):
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
    H10 = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T10 = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y10, yT10 = ytr[tr_sel].astype(np.int32), yte[te_sel].astype(np.int32)

    H100 = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    y100, c100 = d["y"].astype(np.int32), d["coarse"].astype(np.int32)
    T100 = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)[:3000]
    df = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yT100 = df["fine_label"].to_numpy().astype(np.int32)[:3000]
    cT100 = df["coarse_label"].to_numpy().astype(np.int32)[:3000]

    for nm, H, y in [("CIFAR-10", H10, y10), ("CIFAR-100", H100, y100)]:
        cnt = [int((y == c).sum()) for c in np.unique(y)]
        log(f"{nm}: 训练 {H.shape} 类 {len(cnt)} 每类样本 {min(cnt)}~{max(cnt)}")
    log(f"查询：CIFAR-10 {len(T10)}  CIFAR-100 {len(T100)}")

    res = {}

    def build(H, y, nproto_per_class, n_cls):
        pidx = []
        rng = np.random.RandomState(0)
        for c in range(n_cls):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            pidx += list(idx[:nproto_per_class])
        return np.array(pidx)

    def run(tag, H, y, T, yT, n_cls, shard, nnode, npc):
        pidx = build(H, y, npc, n_cls)
        Hp, yp = H[pidx], y[pidx]
        mu, W = fit_W(H, 16)
        Zq = (T - mu) @ W
        nodes = [list(range(n * shard, min((n + 1) * shard, n_cls))) for n in range(nnode)]
        protos, pcls = [], []
        for n, cs in enumerate(nodes):
            for c in cs:
                m = yp == c
                if m.sum() > 0:
                    protos.append(Hp[m].mean(0))
                    pcls.append(c)
        P = np.stack(protos).astype(np.float32)
        pcls = np.array(pcls)
        order = np.argsort(dist(Zq, (P - mu) @ W), 1)
        top1 = float(np.mean(pcls[order[:, 0]] == yT))
        first = first_rank(order, pcls, yT)
        log("\n" + "=" * 72)
        log(f"=== {tag} ===  {len(protos)} 原型（每类 {npc} 张）/ {len(yT)} 查询")
        s = f"  top-1 {top1:.4f}   "
        rr = {}
        for k in [1, 2, 3, 5, 10, 20]:
            v = float(np.mean(first <= k))
            rr[k] = v
            s += f"@{k} {v:.4f}  "
        log(s)
        # 达到 0.95 / 0.99 需要多少候选
        for tgt in [0.95, 0.99]:
            need = next((k for k in range(1, 31) if float(np.mean(first <= k)) >= tgt), None)
            log(f"  达到召回 {tgt}: k = {need if need else '>30'}")
            rr[f"k_for_{tgt}"] = need
        res[tag] = dict(top1=top1, n_proto=len(protos), n_query=int(len(yT)),
                        **{str(k): v for k, v in rr.items()})
        return dict(mu=mu, W=W, Hp=Hp, yp=yp)

    run("A CIFAR-10  5节点×2类", H10, y10, T10, yT10, 10, 2, 5, 500)
    run("B CIFAR-100 50节点×2类", H100, y100, T100, yT100, 100, 2, 50, 500)
    run("C CIFAR-100 20节点×5类", H100, y100, T100, yT100, 100, 5, 20, 500)
    st = run("D CIFAR-100 10节点×10类", H100, y100, T100, yT100, 100, 10, 10, 500)

    # ---- 双层（正常原型）----
    log("\n" + "=" * 72)
    log("=== 双层路由（CIFAR-100，官方 20 超类，每类 500 张原型）===")
    mu, W, Hp, yp = st["mu"], st["W"], st["Hp"], st["yp"]
    # 原型集对应的 coarse 标签
    cp_full = c100  # 训练集 coarse
    pidx = build(H100, y100, 500, 100)
    cp = cp_full[pidx]
    cls_ids = sorted(np.unique(yp))
    Pcls = np.stack([Hp[yp == c].mean(0) for c in cls_ids]).astype(np.float32)
    cls_coarse = np.array([int(cp[yp == c][0]) for c in cls_ids])
    Zp = (Pcls - mu) @ W
    co_ids = sorted(np.unique(cp))
    CP = np.stack([Hp[cp == c].mean(0) for c in co_ids]).astype(np.float32)
    Zq = (T100 - mu) @ W
    co_order = np.argsort(dist(Zq, (CP - mu) @ W), 1)

    for nk in [1, 2, 3, 5, 10]:
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
            if yT100[i] in seen:
                hits += 1
        log(f"  超类@{nk:<2} → 细类top3 召回 {hits/len(Zq):.4f}"
            f"   平均比 {cmp_cnt/len(Zq):.1f} 个原型")
        res[f"hier_nc{nk}"] = dict(recall=hits / len(Zq), compared=cmp_cnt / len(Zq))
    log(f"  （单级 top-3 见上表 D：{res['D CIFAR-100 10节点×10类']['3']:.4f}）")

    with open(os.path.join(ROOT, "results", "results_c100_formal.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
