"""exp11: 1-NN 失败率的解剖 —— 5.6% 到底丢在哪，能不能救回来

问题：路由 hit@1 = 0.944，剩下 5.6% 怎么办？
拆解四层：
  ① 天花板在哪：1-NN 0.947 是全维特征的上界，不是 1.0
  ② top-k 召回：失败样本里正确类是否仍在 top-3 / top-5
  ③ 白捡的修正：kNN 投票能否改善
  ④ 可操作的门控：按 margin 挑出"不确定"的少数查询去扩散，只多花少量请求
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def load_labels_aligned():
    import sys
    sys.path.insert(0, ROOT)
    import exp1_route as E

    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    nc, ni = E.make_shard2(ytr, C["n_nodes"], C["n_classes"], C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    sel = []
    for n, idx in enumerate(ni):
        for k in nc[n]:
            sub = idx[ytr[idx] == k]
            take = rng.choice(sub, size=min(C["per_class_train"], len(sub)), replace=False)
            sel.append(take)
    return ytr[np.concatenate(sel)].astype(np.int32)


def d2mat(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def main():
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    y = load_labels_aligned()
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(H))
    qi, gi = idx[:1000], idx[1000:6000]
    Hq, Hg, yq, yg = H[qi], H[gi], y[qi], y[gi]

    D2 = d2mat(Hq, Hg)
    D = np.sqrt(D2)
    order = np.argsort(D2, axis=1)

    res = {}

    log("=== ① 召回随 k 的变化（1-NN 不是终点）===")
    for k in [1, 2, 3, 5, 10, 20]:
        hit = float(np.mean([yq[i] in yg[order[i, :k]] for i in range(len(yq))]))
        res[f"hit@{k}"] = hit
        log(f"  top-{k:2d} 中至少有一个同类：{hit:.4f}")

    log("\n=== ② kNN 投票能否改善 top-1 ===")
    for k in [3, 5, 9]:
        correct = 0
        for i in range(len(yq)):
            lab = yg[order[i, :k]]
            vals, cnt = np.unique(lab, return_counts=True)
            correct += int(vals[cnt.argmax()] == yq[i])
        acc = correct / len(yq)
        res[f"knn{k}"] = float(acc)
        log(f"  {k}-NN 多数投票准确率：{acc:.4f}")

    d1 = D[np.arange(len(yq)), order[:, 0]]
    d2 = D[np.arange(len(yq)), order[:, 1]]
    margin = d2 - d1
    ratio = d1 / d2
    fail = yg[order[:, 0]] != yq

    log(f"\n=== ③ 失败的 5.6% 长什么样（n={fail.sum()}）===")
    log(f"  失败样本最近邻距离  均值 {d1[fail].mean():.3f}   成功 {d1[~fail].mean():.3f}")
    log(f"  失败样本 margin      均值 {margin[fail].mean():.3f}   成功 {margin[~fail].mean():.3f}")
    log(f"  失败样本 top1/top2 距离比 均值 {ratio[fail].mean():.3f}  成功 {ratio[~fail].mean():.3f}")
    in3 = np.mean([yq[i] in yg[order[i, :3]] for i in np.where(fail)[0]])
    in5 = np.mean([yq[i] in yg[order[i, :5]] for i in np.where(fail)[0]])
    log(f"  → 失败样本里，正确类仍在 top-3 的比例：{in3:.3f}  在 top-5：{in5:.3f}")
    res["fail_total"] = int(fail.sum())
    res["fail_in_top3"] = float(in3)
    res["fail_in_top5"] = float(in5)

    log("\n=== ④ margin 门控：只让最不确定的少数去扩散 ===")
    log(f"{'触发比例':>8} {'总命中率':>10} {'平均请求数':>12} {'救回失败':>10}")
    rows = []
    for p in [5, 10, 20, 30, 50]:
        thr = np.quantile(margin, p / 100.0)
        trig = margin <= thr
        acc = 0.0
        for i in range(len(yq)):
            if trig[i]:
                acc += float(yq[i] in yg[order[i, :3]])
            else:
                acc += float(not fail[i])
        acc /= len(yq)
        req = 1 + 2 * trig.mean()
        saved = (acc - (1 - fail.mean())) / fail.mean()
        rows.append(dict(p=p, acc=float(acc), req=float(req), saved=float(saved)))
        log(f"{p:7d}% {acc:11.4f} {req:12.2f} {saved*100:9.1f}%")

    res["gating"] = rows
    with open(os.path.join(ROOT, "results", "results_failure.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
