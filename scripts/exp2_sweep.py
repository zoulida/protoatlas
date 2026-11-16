# -*- coding: utf-8 -*-
"""
ProtoAtlas 第一阶段验证（含配置扫描）
CIFAR-10 + DINOv2 ViT-S/14 冻结特征 + shard-2 划分 + 网格哈希路由

回答三个问题：
  Q1 本地不认识的样本，能不能路由到见过它的节点上？
  Q2 网格（2 维）筛候选，会损失多少信息？（对比 384 维全扫描上界）
  Q3 维度 d 和网格 G 要取多大？
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
    n_nodes=20,
    n_classes=10,
    shard_per_node=2,
    per_class_train=300,
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
    assert abs(rec - round(rec)) < 1e-9
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
    return node_classes, [np.concatenate(v) for v in node_idx]


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
            t = torch.from_numpy(X[i:i + bs]).float().permute(0, 3, 1, 2) / 255.0
            t = F.interpolate(t, size=CFG["img_size"], mode="bicubic", align_corners=False)
            t = (t - mean) / std
            out[i:i + bs] = phi(t).numpy()
            if i % (bs * 25) == 0:
                el = time.time() - t0
                done = i + min(bs, n - i)
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


def to_cells(z_arr, G):
    c = np.floor(z_arr * G).astype(int)
    return np.clip(c, 0, G - 1)


def neighbors_ring(c, G, r=1):
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx or dy:
                cc = (c[0] + dx, c[1] + dy)
                if 0 <= cc[0] < G and 0 <= cc[1] < G:
                    out.append(cc)
    return out


# ---------------------------------------------------------------- 注册表
def build_registry(H_tr, y_sub, cells_tr, node_classes, node_pos, pos_map, tau):
    reg = defaultdict(list)
    for n in range(len(node_classes)):
        lpos = np.array([pos_map[int(s)] for s in node_pos[n]])
        ys = y_sub[lpos]
        for k in node_classes[n]:
            ml = ys == k
            if ml.sum() == 0:
                continue
            lp = lpos[ml]
            ck = cells_tr[lp]
            for c in np.unique(ck, axis=0):
                m = np.all(ck == c, axis=1)
                sub = lp[m]
                if len(sub) < tau:
                    continue
                reg[(int(c[0]), int(c[1]))].append((n, k, H_tr[sub].mean(0), len(sub)))
    return reg


# ---------------------------------------------------------------- 路由
def route_one(h, z, reg, G, topk=3, ring=0):
    c0 = (int(min(max(z[0] * G, 0), G - 1e-9)), int(min(max(z[1] * G, 0), G - 1e-9)))
    cand = list(reg.get(c0, []))
    used = 1
    if ring > 0 and len(cand) < 3:
        for cc in neighbors_ring(c0, G, ring):
            cand.extend(reg.get(cc, []))
            used += 1
    if not cand:
        return [], used
    best = {}
    for node, k, proto, ncnt in cand:
        d = float(np.dot(h - proto, h - proto))
        if node not in best or d < best[node]:
            best[node] = d
    order = sorted(best.items(), key=lambda kv: kv[1])
    return [n for n, _ in order[:topk]], used


# ---------------------------------------------------------------- 单个配置
def run_config(name, H_tr, H_te, y_tr_sub, y_te_sub, node_classes, node_pos, pos_map,
               dim, grid, tau, seed, ring=0, topk=3):
    t0 = time.time()
    P = fit_coords(H_tr, dim, seed)
    z_tr = apply_coords(P, H_tr)
    z_te = apply_coords(P, H_te)
    cells_tr = to_cells(z_tr, grid)

    reg = build_registry(H_tr, y_tr_sub, cells_tr, node_classes, node_pos, pos_map, tau)
    n_entries = sum(len(v) for v in reg.values())
    cand_counts = [len(set(e[0] for e in v)) for v in reg.values()]
    nodes_reg = {e[0] for v in reg.values() for e in v}

    node_cls_set = [set(cs) for cs in node_classes]
    n_eval = len(H_te)
    h1 = h3 = empty = expd = 0
    for i in range(n_eval):
        top, used = route_one(H_te[i], z_te[i], reg, grid, topk, ring)
        if used > 1:
            expd += 1
        if not top:
            empty += 1
            continue
        y = int(y_te_sub[i])
        if y in node_cls_set[top[0]]:
            h1 += 1
        if any(y in node_cls_set[n] for n in top):
            h3 += 1

    res = dict(
        name=name, dim=dim, grid=grid, tau=tau, ring=ring,
        cells_nonempty=len(reg), entries=n_entries, nodes_registered=len(nodes_reg),
        cand_mean=float(np.mean(cand_counts)) if cand_counts else 0.0,
        cand_max=int(max(cand_counts)) if cand_counts else 0,
        hit1=h1 / n_eval, hit3=h3 / n_eval,
        empty_rate=empty / n_eval, expand_rate=expd / n_eval,
        sec=round(time.time() - t0, 1),
    )
    log(f"  [{name}] d={dim} G={grid} ring={ring}  "
        f"hit@1={res['hit1']:.3f} hit@3={res['hit3']:.3f} "
        f"cand={res['cand_mean']:.1f} empty={res['empty_rate']:.3f} "
        f"格非空={len(reg)}/{grid**2}  ({res['sec']}s)")
    return res


# ---------------------------------------------------------------- 上界：全扫描
def fullscan(H_tr, H_te, y_tr_sub, y_te_sub, node_classes, node_pos, pos_map, topk=3):
    prots, pnode = [], []
    for n in range(len(node_classes)):
        lpos = np.array([pos_map[int(s)] for s in node_pos[n]])
        ys = y_tr_sub[lpos]
        for k in node_classes[n]:
            m = ys == k
            if m.sum() == 0:
                continue
            prots.append(H_tr[lpos[m]].mean(0))
            pnode.append(n)
    Pp = np.stack(prots)
    Pn = np.array(pnode)
    D = np.sqrt(np.maximum(
        (H_te ** 2).sum(1)[:, None] + (Pp ** 2).sum(1)[None, :] - 2 * H_te @ Pp.T, 0))
    order = np.argsort(D, axis=1)
    node_cls_set = [set(cs) for cs in node_classes]
    h1 = h3 = 0
    for i in range(len(H_te)):
        seen, tops = set(), []
        for j in order[i]:
            n = int(Pn[j])
            if n in seen:
                continue
            seen.add(n)
            tops.append(n)
            if len(tops) >= topk:
                break
        y = int(y_te_sub[i])
        if y in node_cls_set[tops[0]]:
            h1 += 1
        if any(y in node_cls_set[n] for n in tops):
            h3 += 1
    return h1 / len(H_te), h3 / len(H_te)


# ---------------------------------------------------------------- 诊断图
def plot_diagnostic(H_tr, y_tr_sub, grid, seed, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P2 = fit_coords(H_tr, 2, seed)
    z2 = apply_coords(P2, H_tr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for k in range(10):
        m = y_tr_sub == k
        ax.scatter(z2[m, 0], z2[m, 1], s=4, alpha=0.55, color=cmap(k), label=str(k))
    for g in range(1, grid):
        ax.axvline(g / grid, color="0.75", lw=0.5)
        ax.axhline(g / grid, color="0.75", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"DINOv2 -> 2D coords (grid {grid}x{grid})")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(markerscale=3, fontsize=8, loc="upper right", framealpha=0.9)

    c = to_cells(z2, grid)
    occ = np.zeros((grid, grid), int)
    for a, b in c:
        occ[a, b] += 1
    ax = axes[1]
    im = ax.imshow(occ.T, origin="lower", cmap="Blues", aspect="equal")
    ax.set_title("samples per cell")
    for a in range(grid):
        for b in range(grid):
            if occ[a, b] > 0:
                ax.text(a, b, str(occ[a, b]), ha="center", va="center",
                        fontsize=max(4, 9 - grid // 4), color="black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log(f"诊断图 → {out_png}")


# ---------------------------------------------------------------- 主流程
def main():
    t_start = time.time()
    log("=" * 78)
    log(f"ProtoAtlas 路由验证  {time.strftime('%Y-%m-%d %H:%M:%S')}  SMOKE={SMOKE}")
    log(f"配置 {CFG}")

    xtr, ytr, xte, yte = load_cifar()
    log(f"数据 train{xtr.shape} test{xte.shape}")

    node_classes, node_idx = make_shard2(ytr, CFG["n_nodes"], CFG["n_classes"],
                                         CFG["shard_per_node"], CFG["seed"])
    cover = defaultdict(list)
    for n, cs in enumerate(node_classes):
        for k in cs:
            cover[k].append(n)
    log(f"shard-2 划分: {CFG['n_nodes']} 节点 × {CFG['shard_per_node']} 类, "
        f"每类覆盖 {[len(cover[k]) for k in range(10)]} 个节点")

    rng = np.random.RandomState(CFG["seed"] + 1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(node_idx):
        for k in node_classes[n]:
            sel = idx[ytr[idx] == k]
            take = rng.choice(sel, size=min(CFG["per_class_train"], len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    te_sel = rng.choice(len(xte), size=min(CFG["n_test"], len(xte)), replace=False)
    log(f"训练侧 {len(tr_sel)} 张, 测试侧 {len(te_sel)} 张")

    sfx = "_smoke" if SMOKE else ""
    H_tr = extract_features(xtr[tr_sel], "train" + sfx)
    H_te = extract_features(xte[te_sel], "test" + sfx)
    y_tr_sub, y_te_sub = ytr[tr_sel], yte[te_sel]

    node_pos = [tr_sel[tr_node == n] for n in range(CFG["n_nodes"])]
    pos_map = {int(s): i for i, s in enumerate(tr_sel)}

    # 上界
    fs1, fs3 = fullscan(H_tr, H_te, y_tr_sub, y_te_sub, node_classes, node_pos, pos_map)
    log("")
    log(f"★ 全扫描上界（384 维，所有节点原型，不用网格）: hit@1={fs1:.3f} hit@3={fs3:.3f}")
    log("")

    # 随机对照
    node_cls_set = [set(cs) for cs in node_classes]
    r1 = r3 = 0
    for i in range(len(te_sel)):
        pick = rng.choice(CFG["n_nodes"], 3, replace=False)
        y = int(y_te_sub[i])
        if y in node_cls_set[pick[0]]:
            r1 += 1
        if any(y in node_cls_set[n] for n in pick):
            r3 += 1
    log(f"随机对照: hit@1={r1/len(te_sel):.3f} hit@3={r3/len(te_sel):.3f}")
    log("")

    configs = [
        ("基准 d2-G8",           2, 8,  0),
        ("细网格 d2-G16",        2, 16, 0),
        ("细网格 d2-G32",        2, 32, 0),
        ("粗网格 d2-G4",         2, 4,  0),
        ("加维 d4-G8",           4, 8,  0),
        ("加维 d8-G8",           8, 8,  0),
        ("加维 d8-G16",          8, 16, 0),
        ("d2-G8 + 邻域多播",     2, 8,  1),
        ("d8-G8 + 邻域多播",     8, 8,  1),
    ]
    log("配置扫描 ...")
    results = []
    for name, dim, grid, ring in configs:
        results.append(run_config(name, H_tr, H_te, y_tr_sub, y_te_sub, node_classes,
                                  node_pos, pos_map, dim, grid, CFG["tau"],
                                  CFG["seed"], ring=ring))

    # 2 维 / 384 维可分性
    from itertools import combinations
    cc = np.stack([H_tr[y_tr_sub == k].mean(0) for k in range(10)])
    intra = np.mean([np.linalg.norm(H_tr[y_tr_sub == k] - cc[k], axis=1).mean() for k in range(10)])
    inter = np.mean([np.linalg.norm(cc[a] - cc[b]) for a, b in combinations(range(10), 2)])
    P2 = fit_coords(H_tr, 2, CFG["seed"])
    z2 = apply_coords(P2, H_tr)
    c2 = np.stack([z2[y_tr_sub == k].mean(0) for k in range(10)])
    intra2 = np.mean([np.linalg.norm(z2[y_tr_sub == k] - c2[k], axis=1).mean() for k in range(10)])
    inter2 = np.mean([np.linalg.norm(c2[a] - c2[b]) for a, b in combinations(range(10), 2)])

    out = dict(cfg=CFG, fullscan_hit1=fs1, fullscan_hit3=fs3,
               random_hit1=r1 / len(te_sel), random_hit3=r3 / len(te_sel),
               sep_384=float(inter / intra), sep_2d=float(inter2 / intra2),
               results=results, elapsed_s=round(time.time() - t_start, 1))
    path = os.path.join(ROOT, "results", "results_stage1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log("")
    log("=" * 78)
    log(f"{'配置':<20}{'hit@1':>8}{'hit@3':>8}{'候选数':>8}{'空路由':>8}{'多播率':>8}")
    log("-" * 60)
    for r in results:
        log(f"{r['name']:<20}{r['hit1']:>8.3f}{r['hit3']:>8.3f}"
            f"{r['cand_mean']:>8.1f}{r['empty_rate']:>8.3f}{r['expand_rate']:>8.3f}")
    log("-" * 60)
    log(f"{'全扫描上界':<20}{fs1:>8.3f}{fs3:>8.3f}")
    log(f"{'随机对照':<20}{r1/len(te_sel):>8.3f}{r3/len(te_sel):>8.3f}")
    log("=" * 78)
    log(f"可分性 384维 类间/类内={inter/intra:.2f}   2维={inter2/intra2:.2f}")
    log(f"总耗时 {out['elapsed_s']}s → {path}")

    try:
        plot_diagnostic(H_tr, y_tr_sub, 8, CFG["seed"],
                        os.path.join(ROOT, "diag_coords.png"))
    except Exception as e:
        log(f"绘图失败: {e}")


if __name__ == "__main__":
    main()
