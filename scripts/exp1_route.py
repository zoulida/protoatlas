# -*- coding: utf-8 -*-
"""
ProtoAtlas 第一阶段验证
CIFAR-10 + DINOv2 ViT-S/14 冻结特征 + shard-2 划分 + 网格哈希路由
核心问题：本地不认识的样本，能不能路由到见过它的节点上？

不涉及 body 训练、不涉及联邦聚合，只测路由本身。
"""
import os, io, json, time
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
LOGS = os.path.join(ROOT, "logs")
for d in (CACHE, LOGS):
    os.makedirs(d, exist_ok=True)
LOG_PATH = os.path.join(LOGS, "run.log")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(s + "\n")


CFG = dict(
    n_nodes=20,           # 20 节点 × 2 类 = 40 槽位 → 每类被 4 个节点覆盖
    n_classes=10,
    shard_per_node=2,
    per_class_train=300,  # 每个 (节点,类) 建原型用的样本数
    n_test=5000,
    dim=2,
    grid=8,
    tau=5,
    seed=0,
    img_size=224,
)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    CFG.update(per_class_train=20, n_test=200, tau=2)


# ---------------------------------------------------------------- 数据
def load_cifar():
    cache = os.path.join(CACHE, "cifar10.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    import pandas as pd
    from PIL import Image

    def rd(split):
        df = pd.read_parquet(os.path.join(DATA, f"{split}.parquet"))
        n = len(df)
        X = np.zeros((n, 32, 32, 3), np.uint8)
        Y = np.zeros(n, np.int64)
        imgs, labs = df["img"].values, df["label"].values
        for i in range(n):
            X[i] = np.array(Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB"))
            Y[i] = labs[i]
        return X, Y

    xtr, ytr = rd("train")
    xte, yte = rd("test")
    np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


# ---------------------------------------------------------------- 划分
def make_shard2(ytr, n_nodes, n_classes, k_per_node, seed=0):
    rng = np.random.RandomState(seed)
    rec = n_nodes * k_per_node / n_classes
    assert abs(rec - round(rec)) < 1e-9, "节点数×每节点类数 必须能被类数整除"
    rec = int(rec)

    for _ in range(5000):
        slots = np.repeat(np.arange(n_classes), rec)
        rng.shuffle(slots)
        assign = slots.reshape(n_nodes, k_per_node)
        if all(len(set(row)) == k_per_node for row in assign):
            break
    else:
        raise RuntimeError("shard 分配失败")

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
    node_idx = [np.concatenate(v) for v in node_idx]
    return node_classes, node_idx


# ---------------------------------------------------------------- 特征
def extract_features(X, tag):
    cache = os.path.join(CACHE, f"feat_{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(8)
    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    n = len(X)
    out = np.zeros((n, 384), np.float32)
    bs, t0 = 64, time.time()
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = X[i:i + bs]
            t = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
            t = F.interpolate(t, size=CFG["img_size"], mode="bicubic", align_corners=False)
            t = (t - mean) / std
            out[i:i + bs] = phi(t).numpy()
            if i % (bs * 25) == 0:
                el = time.time() - t0
                done = i + len(batch)
                log(f"    [{tag}] {done}/{n}  {el:.0f}s  eta {el/max(done,1)*(n-done):.0f}s")
    np.save(cache, out)
    log(f"  [{tag}] 特征完成 {out.shape}  {time.time()-t0:.0f}s")
    return out


# ---------------------------------------------------------------- 坐标
def fit_coords(H_ref, dim, seed=0, nq=1001):
    R = np.random.RandomState(seed).randn(dim, H_ref.shape[1]) / np.sqrt(H_ref.shape[1])
    mu = H_ref.mean(0, keepdims=True)
    raw = (H_ref - mu) @ R.T
    qgrid = np.linspace(0, 1, nq)
    qs = np.stack([np.quantile(raw[:, j], qgrid) for j in range(dim)], axis=1)
    return dict(R=R, mu=mu, qs=qs, qgrid=qgrid)


def apply_coords(P, H):
    raw = (H - P["mu"]) @ P["R"].T
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        z[:, j] = np.interp(raw[:, j], P["qs"][:, j], P["qgrid"])
    return np.clip(z, 0.0, 1.0)


def cell_of(z, G):
    return (int(min(max(z[0] * G, 0), G - 1e-9)), int(min(max(z[1] * G, 0), G - 1e-9)))


def neighbors8(c, G):
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cc = (c[0] + dx, c[1] + dy)
                if 0 <= cc[0] < G and 0 <= cc[1] < G:
                    out.append(cc)
    return out


# ---------------------------------------------------------------- 注册表
def build_registry(H_tr, y_sub, z_tr, node_classes, node_idx, pos_map, G, tau):
    """node_idx[n] = 该节点样本在原始训练集中的索引
       pos_map: 原始索引 -> H_tr 中的行号
       registry[cell] = [(node, cls, proto384, n)]
    """
    reg = defaultdict(list)
    cells_all = np.stack([[int(min(max(z[0] * G, 0), G - 1e-9)),
                           int(min(max(z[1] * G, 0), G - 1e-9))] for z in z_tr])
    for n in range(len(node_classes)):
        lpos = np.array([pos_map[int(s)] for s in node_idx[n]])
        ys = y_sub[lpos]
        for k in node_classes[n]:
            ml = ys == k
            if ml.sum() == 0:
                continue
            lp = lpos[ml]
            ck = cells_all[lp]
            for c in np.unique(ck, axis=0):
                m = np.all(ck == c, axis=1)
                sub = lp[m]
                if len(sub) < tau:
                    continue
                reg[(int(c[0]), int(c[1]))].append((n, k, H_tr[sub].mean(0), len(sub)))
    return reg


# ---------------------------------------------------------------- 路由
def route_one(h, z, reg, G, topk=3):
    c = cell_of(z, G)
    cand = list(reg.get(c, []))
    used = 1
    if not cand:
        for cc in neighbors8(c, G):
            cand.extend(reg.get(cc, []))
            used += 1
            if len(cand) >= 5:
                break
    if not cand:
        return [], used, 0
    best = {}
    for node, k, proto, ncnt in cand:
        d = float(np.dot(h - proto, h - proto))
        if node not in best or d < best[node][0]:
            best[node] = (d, k)
    order = sorted(best.items(), key=lambda kv: kv[1][0])
    return [n for n, _ in order[:topk]], used, len(best)


# ---------------------------------------------------------------- 主流程
def main():
    t_start = time.time()
    log("=" * 72)
    log(f"ProtoAtlas 路由验证  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"配置 {CFG}")

    xtr, ytr, xte, yte = load_cifar()
    log(f"数据 train{xtr.shape} test{xte.shape}")

    node_classes, node_idx = make_shard2(ytr, CFG["n_nodes"], CFG["n_classes"],
                                         CFG["shard_per_node"], CFG["seed"])
    cover = defaultdict(list)
    for n, cs in enumerate(node_classes):
        log(f"  节点{n:02d}: {cs}")
        for k in cs:
            cover[k].append(n)
    log(f"每类覆盖节点数 {dict(sorted((k, len(v)) for k, v in cover.items()))}")

    rng = np.random.RandomState(CFG["seed"] + 1)

    # 训练侧：每 (节点,类) 抽 per_class_train 张
    tr_sel, tr_node = [], []
    for n, idx in enumerate(node_idx):
        for k in node_classes[n]:
            sel = idx[ytr[idx] == k]
            take = rng.choice(sel, size=min(CFG["per_class_train"], len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    log(f"训练侧 {len(tr_sel)} 张（每节点每类 {CFG['per_class_train']}）")

    te_sel = rng.choice(len(xte), size=min(CFG["n_test"], len(xte)), replace=False)
    log(f"测试侧 {len(te_sel)} 张")

    H_tr = extract_features(xtr[tr_sel], "train" + ("_smoke" if SMOKE else ""))
    H_te = extract_features(xte[te_sel], "test" + ("_smoke" if SMOKE else ""))
    y_tr_sub, y_te_sub = ytr[tr_sel], yte[te_sel]

    # 坐标（参考集 = 训练侧全部样本，模拟“公开参照集”）
    P = fit_coords(H_tr, CFG["dim"], CFG["seed"])
    z_tr = apply_coords(P, H_tr)
    z_te = apply_coords(P, H_te)

    # 每节点自己的抽样索引 + 全局位置映射
    node_pos = [tr_sel[tr_node == n] for n in range(CFG["n_nodes"])]
    pos_map = {int(s): i for i, s in enumerate(tr_sel)}

    reg = build_registry(H_tr, y_tr_sub, z_tr, node_classes, node_pos, pos_map,
                         CFG["grid"], CFG["tau"])
    n_entries = sum(len(v) for v in reg.values())
    cand_per_cell = [len(set(e[0] for e in v)) for v in reg.values()]
    nodes_reg = {e[0] for v in reg.values() for e in v}
    log(f"注册表 {len(reg)}/{CFG['grid']**2} 格非空, {n_entries} 条, 涉及 {len(nodes_reg)} 节点")
    log(f"每格候选节点数 均值{np.mean(cand_per_cell):.1f} 中位{np.median(cand_per_cell):.0f} "
        f"最大{max(cand_per_cell)} 最小{min(cand_per_cell)}")

    # ---------------- 路由 ----------------
    node_cls_set = [set(cs) for cs in node_classes]
    n_eval = len(te_sel)
    hits1 = hits3 = empty = expanded = 0
    t0 = time.time()
    for i in range(n_eval):
        top, used, nc = route_one(H_te[i], z_te[i], reg, CFG["grid"], 3)
        if used > 1:
            expanded += 1
        if not top:
            empty += 1
            continue
        y = int(y_te_sub[i])
        if y in node_cls_set[top[0]]:
            hits1 += 1
        if any(y in node_cls_set[n] for n in top):
            hits3 += 1
    log(f"路由评估 {time.time()-t0:.0f}s")

    # ---------------- 对照 1：全扫描（384 维，不用网格）----------------
    prots, pnode, pcls = [], [], []
    for n in range(CFG["n_nodes"]):
        lpos = np.array([pos_map[int(s)] for s in node_pos[n]])
        ys = y_tr_sub[lpos]
        for k in node_classes[n]:
            m = ys == k
            if m.sum() == 0:
                continue
            prots.append(H_tr[lpos[m]].mean(0))
            pnode.append(n)
            pcls.append(k)
    Pp = np.stack(prots)
    Pn = np.array(pnode)
    D = np.sqrt(np.maximum(
        (H_te ** 2).sum(1)[:, None] + (Pp ** 2).sum(1)[None, :] - 2 * H_te @ Pp.T, 0))
    order = np.argsort(D, axis=1)
    fs1 = fs3 = 0
    for i in range(n_eval):
        seen, tops = set(), []
        for j in order[i]:
            n = int(Pn[j])
            if n in seen:
                continue
            seen.add(n)
            tops.append(n)
            if len(tops) >= 3:
                break
        y = int(y_te_sub[i])
        if y in node_cls_set[tops[0]]:
            fs1 += 1
        if any(y in node_cls_set[n] for n in tops):
            fs3 += 1

    # ---------------- 对照 2：随机 ----------------
    rd1 = rd3 = 0
    for i in range(n_eval):
        pick = rng.choice(CFG["n_nodes"], 3, replace=False)
        y = int(y_te_sub[i])
        if y in node_cls_set[pick[0]]:
            rd1 += 1
        if any(y in node_cls_set[n] for n in pick):
            rd3 += 1

    # ---------------- 可分性与负载 ----------------
    from itertools import combinations
    cc = np.stack([H_tr[y_tr_sub == k].mean(0) for k in range(10)])
    intra = np.mean([np.linalg.norm(H_tr[y_tr_sub == k] - cc[k], axis=1).mean() for k in range(10)])
    inter = np.mean([np.linalg.norm(cc[a] - cc[b]) for a, b in combinations(range(10), 2)])
    c2 = np.stack([z_tr[y_tr_sub == k].mean(0) for k in range(10)])
    intra2 = np.mean([np.linalg.norm(z_tr[y_tr_sub == k] - c2[k], axis=1).mean() for k in range(10)])
    inter2 = np.mean([np.linalg.norm(c2[a] - c2[b]) for a, b in combinations(range(10), 2)])

    cidx = np.stack([cell_of(z, CFG["grid"]) for z in z_tr])
    cnts = defaultdict(int)
    for a, b in cidx:
        cnts[(int(a), int(b))] += 1
    occ = np.zeros(CFG["grid"] ** 2, int)
    for (a, b), v in cnts.items():
        occ[a * CFG["grid"] + b] = v
    top10 = np.sort(occ)[::-1][:10].sum() / occ.sum()

    res = dict(
        cfg=CFG,
        n_train=int(len(tr_sel)), n_test=int(n_eval),
        registry_cells=len(reg), registry_entries=n_entries,
        nodes_registered=len(nodes_reg),
        cand_per_cell_mean=float(np.mean(cand_per_cell)),
        cand_per_cell_max=int(max(cand_per_cell)),
        route_hit1=hits1 / n_eval, route_hit3=hits3 / n_eval,
        fullscan_hit1=fs1 / n_eval, fullscan_hit3=fs3 / n_eval,
        random_hit1=rd1 / n_eval, random_hit3=rd3 / n_eval,
        empty_route_rate=empty / n_eval,
        expand_rate=expanded / n_eval,
        sep_384=float(inter / intra), sep_2d=float(inter2 / intra2),
        grid_top10_share=float(top10),
        grid_occupied=int((occ > 0).sum()),
        elapsed_s=float(time.time() - t_start),
    )

    log("")
    log("=" * 72)
    log(f"{'指标':<26}{'值':>12}")
    log("-" * 40)
    for k in ["route_hit1", "route_hit3", "fullscan_hit1", "fullscan_hit3",
              "random_hit1", "random_hit3", "empty_route_rate", "expand_rate",
              "cand_per_cell_mean", "sep_384", "sep_2d", "grid_top10_share"]:
        log(f"{k:<26}{res[k]:>12.4f}")
    log("-" * 40)
    log(f"总耗时 {res['elapsed_s']:.0f}s")

    out = os.path.join(ROOT, "results", "results_stage1.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"结果 → {out}")


if __name__ == "__main__":
    main()
