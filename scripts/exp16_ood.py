"""exp16: 拒识的适用范围 —— 近域 OOD vs 远域 OOD，两种判据

ID    = CIFAR-10 测试集（读作"我的东西"）
OOD-1 = CIFAR-100（100 个从未训练的类，图像域相同 → 最严苛的近域 OOD）
OOD-2 = MNIST（手写数字，画风完全不同 → 远域 OOD）

阈值统一在本节点的 ID 分布上标定为"放行 95%"，再看 OOD 被挡多少。
"""
import os
import io
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
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


def load_raw(name, n_max):
    import pandas as pd
    from PIL import Image

    cache = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"][:n_max]
    fn = {"cifar100": "cifar100_test.parquet", "mnist": "mnist_test.parquet"}[name]
    df = pd.read_parquet(os.path.join(DATA, fn)).iloc[:n_max]
    col = "img" if "img" in df.columns else "image"
    X = np.zeros((len(df), 32, 32, 3), np.uint8)
    for i in range(len(df)):
        im = Image.open(io.BytesIO(df[col].values[i]["bytes"])).convert("RGB")
        if im.size != (32, 32):
            im = im.resize((32, 32), Image.BICUBIC)
        X[i] = np.array(im)
    np.savez_compressed(cache, x=X)
    return X


def feat(X, tag):
    c = os.path.join(CACHE, f"feat_{tag}.npy")
    if os.path.exists(c):
        return np.load(c)
    import torch.nn.functional as F
    from exp1_route import CFG

    torch.set_num_threads(8)
    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    phi.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = np.zeros((len(X), 384), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 64):
            b = torch.from_numpy(X[i:i + 64]).float().permute(0, 3, 1, 2) / 255.0
            b = F.interpolate(b, size=CFG["img_size"], mode="bicubic", align_corners=False)
            out[i:i + 64] = phi((b - mean) / std).numpy()
    np.save(c, out)
    log(f"    [{tag}] 特征完成 {out.shape}")
    return out


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    n_ood = int(os.environ.get("N_OOD", "2000"))
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            protos.append(H[rows].mean(0))
            pcls.append(k)
            pnode.append(n)
    P, pcls, pnode = np.stack(protos).astype(np.float32), np.array(pcls), np.array(pnode)

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    Zp = (P - mu[0]) @ W
    Zq = (T - mu[0]) @ W
    Dq = dist(Zq, Zp)

    log("训练 20 个本地 head，标定阈值（放行 95% 的 ID 样本）...")
    heads, thr_m, thr_g, nidx = {}, {}, {}, {}
    id_m, id_g = {}, {}
    for n in range(C["n_nodes"]):
        m = tr_node == n
        cls = sorted(nc[n])
        heads[n] = train_head(H[m], np.array([cls.index(c) for c in y_tr[m]]))
        idx = [j for j in range(len(P)) if pnode[j] == n]
        nidx[n] = idx
        with torch.no_grad():
            lg = heads[n](torch.from_numpy(T)).numpy()
        mg = lg.max(1) - np.partition(lg, -2, axis=1)[:, -2]
        id_m[n] = mg
        id_g[n] = Dq[:, idx].min(1)
        thr_m[n] = float(np.percentile(mg, 5))
        thr_g[n] = float(np.percentile(id_g[n], 95))

    log(f"\n{'':<12}{'ID 误拒':>10}{'OOD 拦截':>12}{'AUROC':>10}")
    log("-" * 46)

    report = {}
    for name, tag in [("cifar100", "c100"), ("mnist", "mnist")]:
        Xo = load_raw(name, n_ood)
        O = feat(Xo, tag)
        Zo = (O - mu[0]) @ W
        Do = dist(Zo, Zp)
        om, og = [], []
        for n in range(C["n_nodes"]):
            with torch.no_grad():
                lg = heads[n](torch.from_numpy(O)).numpy()
            om.append(lg.max(1) - np.partition(lg, -2, axis=1)[:, -2])
            og.append(Do[:, nidx[n]].min(1))
        OM, OG = np.concatenate(om), np.concatenate(og)
        IM, IG = np.concatenate([id_m[n] for n in range(20)]), \
                 np.concatenate([id_g[n] for n in range(20)])
        # 逐节点比较，避免两组长度不一致
        block_m = float(np.mean([np.mean(om[n] < thr_m[n]) for n in range(20)]))
        block_g = float(np.mean([np.mean(og[n] > thr_g[n]) for n in range(20)]))
        id_frr_m = float(np.mean([np.mean(id_m[n] < thr_m[n]) for n in range(20)]))
        id_frr_g = float(np.mean([np.mean(id_g[n] > thr_g[n]) for n in range(20)]))
        log(f"{name:<12}{'':>10}{block_m:>12.3f}{auroc(IM, OM):>10.3f}   ← margin 判据")
        log(f"{'':<12}{'':>10}{block_g:>12.3f}{auroc(-IG, -OG):>10.3f}   ← 距离判据")
        log(f"{'':<12}ID 误拒 margin {id_frr_m:.3f} / geo {id_frr_g:.3f}")
        report[name] = dict(block_margin=block_m, block_geo=block_g,
                            auroc_margin=auroc(IM, OM), auroc_geo=auroc(-IG, -OG),
                            id_frr_margin=id_frr_m, id_frr_geo=id_frr_g)
        log("-" * 46)

    with open(os.path.join(ROOT, "results", "results_ood.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
