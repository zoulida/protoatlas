"""exp15: 拒识在被问到"真·陌生类"时管不管用
ID  = CIFAR-10 测试集（10 个已知类）
OOD = CIFAR-100 测试集（100 个从未出现过的类，图像域相同，属最严苛的近域 OOD）

回答两件事：
① 判据能不能把 OOD 认出来（AUROC）
② OOD 请求进来后，多少被拒识、多少被硬答成某个已知类
"""
import os
import io
import sys
import json
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def auroc(score_in, score_out):
    s = np.concatenate([score_in, score_out])
    y = np.concatenate([np.ones(len(score_in)), np.zeros(len(score_out))])
    order = np.argsort(s)
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_c100(n_max=None):
    cache = os.path.join(CACHE, "c100.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        X, Y = d["x"], d["y"]
        return X[:n_max] if n_max else X, Y[:n_max] if n_max else Y
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(os.path.join(DATA, "cifar100_test.parquet"))
    if n_max:
        df = df.iloc[:n_max]
    n = len(df)
    X = np.zeros((n, 32, 32, 3), np.uint8)
    Y = np.zeros(n, np.int64)
    imgs, labs = df["img"].values, df["fine_label"].values
    for i in range(n):
        X[i] = np.array(Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB"))
        Y[i] = labs[i]
    np.savez_compressed(cache, x=X, y=Y)
    return X, Y


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
    bs, t0 = 64, time.time()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.from_numpy(X[i:i + bs]).float().permute(0, 3, 1, 2) / 255.0
            b = F.interpolate(b, size=CFG["img_size"], mode="bicubic", align_corners=False)
            out[i:i + bs] = phi((b - mean) / std).numpy()
            if i % (bs * 25) == 0:
                el = time.time() - t0
                log(f"    [{tag}] {i+len(b)}/{len(X)} {el:.0f}s")
    np.save(c, out)
    return out


def main():
    from exp12_proto_route import build_split
    from exp14_reject_head import train_head

    n_max = int(os.environ.get("N_OOD", "3000"))
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    NAME = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]

    log(f"读取 CIFAR-100 测试集（最多 {n_max} 张，100 个陌生类）...")
    Xo, Yo = load_c100(n_max)
    O = feat(Xo, "c100")
    log(f"OOD 特征 {O.shape}")

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            if rows.sum() == 0:
                continue
            protos.append(H[rows].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zq, Zp = (T - mu) @ W, (P - mu) @ W
    Zo = (O - mu[0]) @ W

    def dist(A, B):
        return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                                  - 2.0 * (A @ B.T), 0))

    D = dist(Zq, Zp)
    Do = dist(Zo, Zp)

    # 训练 20 个 head，并取 ID margin 的 5% 分位作阈值（放行 95%）
    log("训练 20 个本地 head 并标定阈值...")
    heads, thr = {}, {}
    id_scores, ood_scores = [], []
    id_geo, ood_geo = [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        cls = sorted(nc[n])
        Xn = H[m]
        yn = np.array([cls.index(c) for c in y_tr[m]])
        heads[n] = train_head(Xn, yn)
        idx = [j for j in range(len(P)) if pnode[j] == n]
        med = np.median([np.median(D[:, j]) for j in idx])
        with torch.no_grad():
            lg_i = heads[n](torch.from_numpy(T)).numpy()
            lg_o = heads[n](torch.from_numpy(O)).numpy()
        mi = lg_i.max(1) - np.partition(lg_i, -2, axis=1)[:, -2]
        mo = lg_o.max(1) - np.partition(lg_o, -2, axis=1)[:, -2]
        thr[n] = float(np.percentile(mi, 5))
        id_scores.append(mi)
        ood_scores.append(mo)
        id_geo.append(-D[:, idx].min(1) / max(med, 1e-9))
        ood_geo.append(-Do[:, idx].min(1) / max(med, 1e-9))

    ID = np.concatenate(id_scores)
    OD = np.concatenate(ood_scores)
    IDg = np.concatenate(id_geo)
    ODg = np.concatenate(ood_geo)

    log("\n" + "=" * 78)
    log("① 判据对'真·陌生类'的区分能力（20 节点 × 全部样本）")
    log("=" * 78)
    log(f"  margin 判据：ID 中位 {np.median(ID):.1f}  OOD 中位 {np.median(OD):.1f}  "
        f"AUROC = {auroc(ID, OD):.4f}")
    log(f"  几何判据：AUROC = {auroc(IDg, ODg):.4f}")
    log(f"  在 5% 误拒率下，OOD 拦截率："
        f"margin {np.mean(OD < np.percentile(ID, 5)):.4f} / "
        f"geo {np.mean(ODg < np.percentile(IDg, 5)):.4f}")

    # ② OOD 请求端到端：走路由 → top1 节点判定
    log("\n" + "=" * 78)
    log("② 陌生请求进来后会发生什么（走真实路由）")
    log("=" * 78)
    order_o = np.argsort(Do, axis=1)
    for mode in ["none", "head", "geo"]:
        rej = acc = 0
        for i in range(len(O)):
            j = order_o[i, 0]
            n, c = pnode[j], pcls[j]
            if mode == "none":
                accept = True
            elif mode == "head":
                with torch.no_grad():
                    lg = heads[n](torch.from_numpy(O[i][None, :])).numpy()[0]
                mg = lg.max() - np.partition(lg, -2)[-2]
                accept = mg >= thr[n]
            else:
                idx = [jj for jj in range(len(P)) if pnode[jj] == n]
                med = np.median([np.median(D[:, jj]) for jj in idx])
                accept = Do[i, idx].min() / max(med, 1e-9) <= np.percentile(
                    [D[:, jj].min() / max(med, 1e-9) for jj in idx], 95)
            if accept:
                acc += 1
            else:
                rej += 1
        log(f"  [{mode:<5}] 拒识 {rej/len(O):.4f}   被硬答成某个已知类 {acc/len(O):.4f}")

    # ③ 同时看 ID 侧的误拒（不能为了挡 OOD 把自家样本全拒了）
    log("\n" + "=" * 78)
    log("③ 同一个阈值在 ID 侧的表现")
    log("=" * 78)
    log(f"  margin 阈值（各节点 ID margin 的 5% 分位）→ ID 误拒率 "
        f"{np.mean([np.mean(s < thr[n]) for n, s in enumerate(id_scores)]):.4f}")


if __name__ == "__main__":
    main()
