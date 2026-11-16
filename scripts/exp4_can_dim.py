# -*- coding: utf-8 -*-
"""
exp4_can_dim.py  ——  CAN 到底需要多少维？

先澄清一点：CAN 只在节点加入时划分空间，格数等于节点数 m，不存在 G^d 爆炸。
这一判断成立，因此本脚本验证真正的约束在哪里。

A 部分：纯模拟 CAN 的递归二分（体积最大优先 / 最长边切），
        统计 m 个 zone 实际"切开"了多少个维度。
B 部分：用真实 DINOv2 特征，比较随机投影 / PCA 后，
        前 log2(m) 个维度各承载多少判别信息。
"""
import json
import numpy as np

RNG = np.random.RandomState(0)


# ============================================================
# A. CAN 分区模拟
# ============================================================
def simulate_can(d, m, strategy="longest"):
    """从整个 [0,1]^d 开始，每次切体积最大的 zone，沿最长轴二分。"""
    lo = [np.zeros(d)]
    hi = [np.ones(d)]
    while len(lo) < m:
        vols = np.array([np.prod(hi[i] - lo[i]) for i in range(len(lo))])
        i = int(np.argmax(vols))
        axis = int(np.argmax(hi[i] - lo[i]))       # 最长轴
        mid = (lo[i][axis] + hi[i][axis]) / 2.0

        lo_new = lo[i].copy()
        hi_new = hi[i].copy()
        hi[i][axis] = mid                 # 老 zone 收缩
        lo_new[axis] = mid                # 新 zone 接管另一半
        lo.append(lo_new)
        hi.append(hi_new)
    return np.array(lo), np.array(hi)


def analyze(d, m):
    lo, hi = simulate_can(d, m)
    side = hi - lo                                    # (m, d) 每 zone 每维的长度

    # 每维的全局分段数：该维上出现过的不同 lo 值的个数
    seg = np.zeros(d, dtype=int)
    for j in range(d):
        seg[j] = len(np.unique(np.round(lo[:, j], 9)))

    touched = int(np.sum(seg > 1))                    # 真正被切开的维度数
    # 每 zone 自己的有效边数（该维长度 < 1 的个数）
    per_zone_cut = np.sum(side < 1 - 1e-9, axis=1)

    # 两两 zone 在某一维上完全重合的概率
    # 抽样 2000 对估计
    idx = RNG.randint(0, m, size=(2000, 2))
    same = (np.abs(lo[idx[:, 0]] - lo[idx[:, 1]]) < 1e-9) & \
           (np.abs(hi[idx[:, 0]] - hi[idx[:, 1]]) < 1e-9)
    fully_overlap = float(np.mean(np.all(same, axis=1)))     # 两个 zone 完全相同
    dim_overlap = same.mean(axis=0)                          # 每维上重合的比例

    return dict(
        d=d, m=m,
        zones=len(lo),
        touched_dims=touched,
        utilization=touched / d,
        per_zone_cut_mean=float(per_zone_cut.mean()),
        per_zone_cut_max=int(per_zone_cut.max()),
        side_mean=float(side.mean()),
        side_p50=float(np.median(side)),
        dims_untouched_ratio=float(np.mean(seg == 1)),
        fully_overlap_ratio=fully_overlap,
        seg_nonzero=seg[seg > 1].tolist()[:20],
    )


print("=" * 78)
print("A. CAN 递归二分：m 个 zone 实际撕开了几个维度")
print("=" * 78)
print(f"{'d':>5} {'m':>6} {'zone数':>7} {'被切维度':>9} {'利用率':>8} "
      f"{'每zone刀数':>11} {'平均边长':>10} {'空闲维度比':>10}")
rows_a = []
for d in [2, 3, 4, 8, 10, 16, 64, 384]:
    for m in [100, 1000]:
        r = analyze(d, m)
        rows_a.append(r)
        print(f"{r['d']:>5} {r['m']:>6} {r['zones']:>7} {r['touched_dims']:>9} "
              f"{r['utilization']:>7.1%} {r['per_zone_cut_mean']:>11.1f} "
              f"{r['side_mean']:>10.3f} {r['dims_untouched_ratio']:>9.1%}")

print()
print("关键对照：log2(m) 与 被切维度数")
for m in [100, 1000]:
    r = [x for x in rows_a if x["d"] == 384 and x["m"] == m][0]
    print(f"  m={m:>5}: log2(m)={np.log2(m):5.2f}   "
          f"实测被切维度={r['touched_dims']}   "
          f"每 zone 平均切了 {r['per_zone_cut_mean']:.1f} 刀")


# ============================================================
# B. 真实特征：判别信息集中在几个维度上？
# ============================================================
print()
print("=" * 78)
print("B. DINOv2 真实特征：前 k 维承载多少判别信息")
print("=" * 78)

H = np.load(os.path.join(ROOT, "cache", "feat_train.npy")).astype(np.float64)

# 复现 exp1 的抽样索引，取到与 H 对应的标签（N_NODES/PER_CLS 必须与 exp1 一致）
N_NODES, N_CLASSES, K_PER, PER_CLS, SEED = 20, 10, 2, 300, 0


def make_shard2(ytr, n_nodes, n_classes, k_per_node, seed=0):
    rng = np.random.RandomState(seed)
    rec = int(n_nodes * k_per_node / n_classes)
    for _ in range(5000):
        slots = np.repeat(np.arange(n_classes), rec)
        rng.shuffle(slots)
        assign = slots.reshape(n_nodes, k_per_node)
        if all(len(set(row)) == k_per_node for row in assign):
            break
    node_classes = [list(map(int, r)) for r in assign]
    idx_by_class = {k: np.where(ytr == k)[0] for k in range(n_classes)}
    node_idx = [[] for _ in range(n_nodes)]
    for k in range(n_classes):
        owners = [n for n in range(n_nodes) if k in node_classes[n]]
        rng.shuffle(owners)
        idx = idx_by_class[k].copy()
        rng.shuffle(idx)
        for n, ch in zip(owners, np.array_split(idx, len(owners))):
            node_idx[n].append(ch)
    return node_classes, [np.concatenate(v) for v in node_idx]


z = np.load(os.path.join(ROOT, "cache", "cifar10.npz"), allow_pickle=True)
ytr = z["ytr"]
node_classes, node_idx = make_shard2(ytr, N_NODES, N_CLASSES, K_PER, SEED)
rng = np.random.RandomState(SEED + 1)
tr_sel = []
for n, idx in enumerate(node_idx):
    for k in node_classes[n]:
        sel = idx[ytr[idx] == k]
        take = rng.choice(sel, size=min(PER_CLS, len(sel)), replace=False)
        tr_sel.append(take)
tr_sel = np.concatenate(tr_sel)
y_tr = ytr[tr_sel]
print(f"特征矩阵 {H.shape}   标签 {y_tr.shape}   类别 {len(np.unique(y_tr))}")
assert len(y_tr) == len(H), "抽样索引与缓存特征条数不符"

if True:
    Mu = H.mean(0)
    Hc = H - Mu
    # PCA
    C = Hc.T @ Hc / len(Hc)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    ev = w / w.sum()
    Z_pca = Hc @ V[:, :16]

    # 类中心方差（判别信息的粗代理）
    cls = np.unique(y_tr)
    ctr = np.stack([Hc[y_tr == c].mean(0) for c in cls])
    # 投影到各主成分后，类中心在这个方向上的方差
    ctr_proj = ctr @ V[:, :16]
    cls_var = ctr_proj.var(axis=0)

    print(f"{'前 k 主成分':>12} {'累计方差占比':>14} {'类间方差占比':>14}")
    for k in [1, 2, 4, 8, 10, 16]:
        print(f"{k:>12} {ev[:k].sum():>13.1%} "
              f"{cls_var[:k].sum() / cls_var.sum():>13.1%}")

    # 随机投影对照
    Rr = np.random.RandomState(0).randn(384, 16)
    Z_rnd = Hc @ Rr
    rnd_var = ( (ctr @ Rr).var(axis=0) )
    print(f"\n随机投影前 10 维承载的类间方差占比: "
          f"{rnd_var[:10].sum() / rnd_var.sum():.1%}")
    print(f"PCA    前 10 维承载的类间方差占比: "
          f"{cls_var[:10].sum() / cls_var.sum():.1%}")

    # 单维判别力：每维的 Fisher 比（类间/类内）
    def fisher_ratio(proj):
        k = proj.shape[1]
        out = []
        for j in range(k):
            x = proj[:, j]
            mu = x.mean()
            sb = sum(len(x[y_tr == c]) * (x[y_tr == c].mean() - mu) ** 2
                     for c in cls) / len(x)
            sw = sum(((x[y_tr == c] - x[y_tr == c].mean()) ** 2).sum()
                     for c in cls) / len(x)
            out.append(sb / (sw + 1e-12))
        return np.array(out)

    f_pca = fisher_ratio(Z_pca)
    f_rnd = fisher_ratio(Z_rnd)
    print(f"\nPCA  前10维 Fisher 比之和 = {f_pca[:10].sum():.4f}")
    print(f"随机 前10维 Fisher 比之和 = {f_rnd[:10].sum():.4f}")
    print(f"比值 = {f_pca[:10].sum() / (f_rnd[:10].sum() + 1e-12):.1f}x")

    # 全部 384 个主成分上的 Fisher 比 → 前 10 维占全部判别力的比例
    f_all = fisher_ratio(Hc @ V)
    tot = f_all.sum()
    print(f"\n全部 384 维 Fisher 比之和 = {tot:.4f}")
    print(f"PCA 前 10 维占全部判别力 = {f_pca[:10].sum() / tot:.1%}")
    print(f"原始空间任取 10 维的期望占比 = {10 / 384:.1%}")
    print(f"→ 倍数 = {(f_pca[:10].sum() / tot) / (10 / 384):.1f}x")

out = dict(rows_a=rows_a)
with open(os.path.join(ROOT, "results", "results_can_dim.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n→ results_can_dim.json")
