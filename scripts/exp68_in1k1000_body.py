"""exp68: 千类尺度（K=1000）下，body 是否有用

此前的 body 实验都在 100 类以内：CIFAR-100（每节点 20 类）、ImageNet-100
（每节点 20 类）。本脚本把类数抬到 1000，节点 50 个、每节点 20 类，vnode
总数 1000，与第 5.4 节千类档同规模。

数据：ImageNet-1k 验证集特征（cache/feat_in1k.npy，每类 50 张）。
  每类前 25 张 → 训练，后 25 张 → 测试（25000 张）
样本仍受验证集限制（每节点 20×25=500 张），故这一档只回答「类数到千级、
任务足够难时 body 有没有用」，样本量的作用由 exp67 单独回答。

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

K = 1000
N_NODES = 50
N_TR = 25
D_PROJ = 100


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


def main():
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    tr_idx, te_idx = [], []
    for c in range(K):
        rows = np.where(yt == c)[0]
        tr_idx.append(rows[:N_TR])
        te_idx.append(rows[N_TR:])
    H, y = Ht[np.concatenate(tr_idx)], yt[np.concatenate(tr_idx)]
    T, yT = Ht[np.concatenate(te_idx)], yt[np.concatenate(te_idx)]
    log(f"训练 {H.shape}，测试 {T.shape}，{N_NODES} 节点 × {K//N_NODES} 类")

    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(K), N_NODES))}

    mu, W, d_eff = fit_lda(H, y, D_PROJ)
    log(f"判别投影 d={d_eff}")

    protos, pcls, pnode = [], [], []
    for n in range(N_NODES):
        for c in cls_of_node[n]:
            protos.append(H[y == c].mean(0))
            pcls.append(c)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zq, Zp = (T - mu) @ W, (P - mu) @ W
    log("算距离矩阵…")
    Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                            - 2 * Zq @ Zp.T, 0))
    order = np.argpartition(Dm, 12, axis=1)[:, :13]
    # 逐行按真实距离排序，取 top-1（按类去重对该行通常不改变第一项）
    top_node = np.zeros(len(T), np.int32)
    top_cls = np.zeros(len(T), np.int32)
    for i in range(len(T)):
        o = order[i][np.argsort(Dm[i, order[i]])]
        top_cls[i] = int(pcls[o[0]])
        top_node[i] = int(pnode[o[0]])
    route = float(np.mean(top_cls == yT))
    log(f"top-1 路由精度 {route:.4f}")

    Hn, yn = {}, {}
    for n in range(N_NODES):
        m = np.isin(y, cls_of_node[n])
        Hn[n] = H[m]
        yn[n] = np.array([cls_of_node[n].index(c) for c in y[m]], dtype=np.int64)

    res = {"route_top1": route, "d": int(d_eff)}

    def evaluate(bodies, heads):
        right = 0
        for n in range(N_NODES):
            m = top_node == n
            if m.sum() == 0:
                continue
            pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T[m]).argmax(1)]
            right += int((pred == yT[m]).sum())
        return right / len(T)

    for tag, hid in (("E0", None), ("E1", 256)):
        bodies, heads = {}, {}
        for n in range(N_NODES):
            bodies[n], heads[n] = build(hid, K // N_NODES)
            train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
        res[tag] = evaluate(bodies, heads)
        log(f"  {tag} D1 {res[tag]:.4f}")
    res["gain_pt"] = round((res["E1"] - res["E0"]) * 100, 2)
    log(f"增益 {res['gain_pt']:+.2f} 点")

    with open(os.path.join(ROOT, "results", "results_in1k1000_body.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    log("写入 results_in1k1000_body.json")


if __name__ == "__main__":
    main()
