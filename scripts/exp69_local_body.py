"""exp69: body 对「本地判别」本身的提升有多大（区别于端到端）

起因：作者问「本文的 body 是本地训练的，只能提升本地精度，那训练 body 对
本地分类有提升吗」。此前的实验只报了端到端 D1，本脚本把两层拆开报：

  local  各节点在自己负责的类上、对测试集的对应样本作答的准确率
         （即该节点作为「本地分类器」的精度，与路由无关）
  D1     端到端：请求按 top-1 路由送达后，宿主节点作答的准确率

四档（类数 / 每节点类数 / 每节点样本数都不同）：
  CIFAR-100        100 类，20 节点 × 5 类，每类 500 张
  ImageNet-100 N=1 100 类，5 节点 × 20 类，每类 25 张
  ImageNet-100 N=8 同上，每类 25×8 张（增广）
  ImageNet-1000    1000 类，50 节点 × 20 类，每类 25 张

每档比 E0（body 取恒等）与 E1（body = 384→256→ReLU，与 head 联合训练）。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp61_body_head import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


def log(*a):
    print(*a, flush=True)


def build(hid, kin, d=384):
    if hid is None:
        return nn.Identity(), nn.Linear(d, kin)
    return nn.Sequential(nn.Linear(d, hid), nn.ReLU()), nn.Linear(hid, kin)


def train_join(X, y, body, head, epochs, lr=5e-3, wd=1e-4):
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()),
                           lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(body(Xt)), yt).backward()
        opt.step()


def infer(body, head, X, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(head(body(torch.from_numpy(X[i:i + bs]))).numpy())
    return np.concatenate(out, 0)


def run(tag, H, y, T, yT, n_nodes, epochs, d_proj=None):
    classes = np.unique(y)
    K = len(classes)
    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(classes, n_nodes))}
    log(f"\n=== {tag}：{K} 类，{n_nodes} 节点 × {K//n_nodes} 类，"
        f"训练 {len(H)} 张（每节点 {len(H)//n_nodes} 张），测试 {len(T)} 张 ===")

    mu, W, d_eff = fit_lda(H, y, min(d_proj or K - 1, 384))
    protos, pcls, pnode = [], [], []
    for n in range(n_nodes):
        for c in cls_of_node[n]:
            protos.append(H[y == c].mean(0))
            pcls.append(c)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zq, Zp = (T - mu) @ W, (P - mu) @ W
    Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                            - 2 * Zq @ Zp.T, 0))
    top_node = np.array([int(pnode[j]) for j in np.argmin(Dm, 1)])
    top_cls = np.array([int(pcls[j]) for j in np.argmin(Dm, 1)])
    route = float(np.mean(top_cls == yT))

    Hn, yn, rows_of = {}, {}, {}
    for n in range(n_nodes):
        m = np.isin(y, cls_of_node[n])
        Hn[n] = H[m]
        yn[n] = np.array([cls_of_node[n].index(c) for c in y[m]], dtype=np.int64)
        rows_of[n] = np.isin(yT, cls_of_node[n])
    log(f"路由精度 {route:.4f}；各节点负责的测试样本 {[int(rows_of[n].sum()) for n in range(min(5, n_nodes))]}…")

    out = {"dataset": tag, "route_top1": route}
    for cfg, hid in (("E0", None), ("E1", 256)):
        bodies, heads = {}, {}
        for n in range(n_nodes):
            bodies[n], heads[n] = build(hid, K // n_nodes)
            train_join(Hn[n], yn[n], bodies[n], heads[n], epochs)
        loc_right = loc_tot = d1_right = 0
        for n in range(n_nodes):
            m = rows_of[n]
            if m.sum() == 0:
                continue
            pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T[m]).argmax(1)]
            loc_right += int((pred == yT[m]).sum())
            loc_tot += int(m.sum())
        for n in range(n_nodes):
            m = top_node == n
            if m.sum() == 0:
                continue
            pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T[m]).argmax(1)]
            d1_right += int((pred == yT[m]).sum())
        out[cfg] = {"local": loc_right / loc_tot, "D1": d1_right / len(T)}
        log(f"  {cfg}  本地判别 {out[cfg]['local']:.4f}   端到端 {out[cfg]['D1']:.4f}")

    out["local_gain_pt"] = round((out["E1"]["local"] - out["E0"]["local"]) * 100, 2)
    out["e2e_gain_pt"] = round((out["E1"]["D1"] - out["E0"]["D1"]) * 100, 2)
    log(f"  增益：本地判别 {out['local_gain_pt']:+.2f} 点，端到端 {out['e2e_gain_pt']:+.2f} 点")
    return out


def main():
    res = {}

    # ---------- CIFAR-100：100 类，20 节点 × 5 类
    H, y, T, yT = load_c100()
    res["c100"] = run("CIFAR-100", H, y, T[:3000], yT[:3000], 20, 400)

    # ---------- ImageNet-100：5 节点 × 20 类
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    te_idx = np.concatenate([np.where(yt == c)[0][25:50] for c in range(100)])
    T1, yT1 = Ht[te_idx], yt[te_idx]
    for N in (1, 8):
        f = os.path.join(CACHE, f"feat_in1k100_aug{N}.npy")
        if not os.path.exists(f):
            continue
        H1 = np.load(f).astype(np.float32)
        y1 = np.load(os.path.join(CACHE, f"in1k100_aug{N}_y.npy")).astype(np.int64)
        res[f"in1k100_N{N}"] = run(f"ImageNet-100 (N={N})", H1, y1, T1, yT1, 5, 300)

    # ---------- ImageNet-1000：50 节点 × 20 类
    tr_idx = np.concatenate([np.where(yt == c)[0][:25] for c in range(1000)])
    te_idx = np.concatenate([np.where(yt == c)[0][25:] for c in range(1000)])
    res["in1k1000"] = run("ImageNet-1000", Ht[tr_idx], yt[tr_idx],
                          Ht[te_idx], yt[te_idx], 50, 400, d_proj=100)

    with open(os.path.join(ROOT, "results", "results_local_body.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n写入 results_local_body.json")


if __name__ == "__main__":
    main()
