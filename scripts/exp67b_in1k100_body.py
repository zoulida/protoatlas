"""exp67b: ImageNet-100 类上，样本量（增广倍数）对 body 增益的影响

配合 exp67a：训练特征按每类 25 张 × N 增广给出（cache/feat_in1k100_aug{N}.npy），
测试取 feat_in1k 里每类后 25 张（未增广，与训练图不重叠）。

设定：100 类，5 节点 × 20 类；坐标由判别投影给出，vnode = (节点,类)。
路由与候选在各档完全相同（body 不进路由），只改本地判别器：
  E0  body 取恒等，head 直接接 384 维
  E1  body = 384→256→ReLU，与 head 联合训练
主指标 D1 = top-1 候选的宿主节点作答后的端到端精度。
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

torch.manual_seed(0)
np.random.seed(0)

NCLS = 100
N_NODES = 5
N_TR = 25
EPOCHS = {1: 600, 4: 400, 8: 300, 16: 200}


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


def infer(body, head, X):
    with torch.no_grad():
        return head(body(torch.from_numpy(X))).numpy()


def main():
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    keep = yt < NCLS
    Ht, yt = Ht[keep], yt[keep]
    # 每类后 25 张作测试
    te_idx = []
    for c in range(NCLS):
        rows = np.where(yt == c)[0]
        te_idx.append(rows[N_TR:N_TR + 25])
    te_idx = np.concatenate(te_idx)
    T, yT = Ht[te_idx], yt[te_idx]
    log(f"测试 {T.shape}，每类 {len(te_idx)//NCLS} 张")

    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(NCLS), N_NODES))}
    res = {}

    for N in [int(x) for x in sys.argv[1:]] or [1, 8]:
        f = os.path.join(CACHE, f"feat_in1k100_aug{N}.npy")
        fy = os.path.join(CACHE, f"in1k100_aug{N}_y.npy")
        if not os.path.exists(f):
            log(f"缺 {f}，跳过 N={N}")
            continue
        H, y = np.load(f).astype(np.float32), np.load(fy).astype(np.int64)
        log(f"\n=== N={N}，训练 {H.shape}，每节点 {len(H)//N_NODES} 张 ===")

        # 自检：aug=0 的特征应与 feat_in1k 的同类前 25 张一致
        same = (y == 0)
        cos = float(np.mean((H[same][:25] * Ht[yt == 0][:25]).sum(1) /
                            (np.linalg.norm(H[same][:25], axis=1) *
                             np.linalg.norm(Ht[yt == 0][:25], axis=1))))
        log(f"自检：aug=0 与原图特征的平均余弦 {cos:.4f}")

        mu, W, d_eff = fit_lda(H, y, NCLS - 1)
        protos, pcls, pnode = [], [], []
        for n in range(N_NODES):
            for c in cls_of_node[n]:
                protos.append(H[y == c].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zq, Zp = (T - mu) @ W, (P - mu) @ W
        Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                                - 2 * Zq @ Zp.T, 0))
        order = np.argsort(Dm, 1)
        top_node = np.zeros(len(T), np.int32)
        top_cls = np.zeros(len(T), np.int32)
        for i in range(len(T)):
            seen = set()
            for j in order[i]:
                c = int(pcls[j])
                if c in seen:
                    continue
                seen.add(c)
                top_cls[i] = c
                top_node[i] = int(pnode[j])
                break
        route = float(np.mean(top_cls == yT))
        log(f"判别投影 d={d_eff}，top-1 路由精度 {route:.4f}")

        Hn, yn = {}, {}
        for n in range(N_NODES):
            m = np.isin(y, cls_of_node[n])
            Hn[n] = H[m]
            yn[n] = np.array([cls_of_node[n].index(c) for c in y[m]], dtype=np.int64)

        out = {"N": N, "samples_per_node": int(len(H) / N_NODES), "route_top1": route}

        def evaluate(bodies, heads):
            right = 0
            for n in range(N_NODES):
                pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
                m = top_node == n
                right += int((pred[m] == yT[m]).sum())
            return right / len(T)

        for tag, hid in (("E0", None), ("E1", 256)):
            bodies, heads = {}, {}
            for n in range(N_NODES):
                bodies[n], heads[n] = build(hid, NCLS // N_NODES)
                train_join(Hn[n], yn[n], bodies[n], heads[n], EPOCHS.get(N, 300))
            out[tag] = evaluate(bodies, heads)
            log(f"  {tag} D1 {out[tag]:.4f}")
        out["gain_pt"] = round((out["E1"] - out["E0"]) * 100, 2)
        log(f"  增益 {out['gain_pt']:+.2f} 点")
        res[str(N)] = out

    with open(os.path.join(ROOT, "results", "results_in1k100_body.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    log("\n写入 results_in1k100_body.json")


if __name__ == "__main__":
    main()
