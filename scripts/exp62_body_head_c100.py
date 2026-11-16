"""exp62: CIFAR-100 上放开 body —— body 与 head 分开训练对端到端精度的影响

设定与 exp53 对齐：100 类、200 节点 × 2 类、每类 4 副本、每节点 2×125=250 张、
测试 3000 张、判别投影 d=99、vnode 每 (节点,类) 一个、全局排序按类去重取最近一类。

判别器三档（路由完全相同，只换本地判别器）：
  E0  body 取恒等（正文现读数），head 直接接 384 维
  E1  body = 384→256→ReLU，本地联合训练，不同步
  E2  body 同 E1，在坐标最近的 8 个节点间同步 3 轮（模拟邻近共同训练）

坐标只由协议资产 φ 决定，body 不进入路由，故候选列表三档完全一致。
"""
import os
import sys
import time
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp32_end2end_all_lda import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100, build_split, N_TEST  # noqa: E402

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


def infer(body, head, X):
    with torch.no_grad():
        return head(body(torch.from_numpy(X))).numpy()


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    nc, tr_sel, tr_node = build_split(y)
    Htr, ytr = H[tr_sel], y[tr_sel]
    Nn = len(nc)
    log(f"数据：节点 {Nn} 训练 {Htr.shape} 测试 {T.shape}")

    mu, W, d_eff = fit_lda(Htr, ytr, 99)
    log(f"判别投影 d={d_eff}")

    # ---------- 路由：全局排序、按类去重取最近一类（与 exp53 同口径）
    protos, pcls, pnode, node_z = [], [], [], {}
    for n in range(Nn):
        m = tr_node == n
        zs = []
        for c in nc[n]:
            X = Htr[m & (ytr == c)]
            if len(X) == 0:
                continue
            p = X.mean(0)
            protos.append(p)
            pcls.append(c)
            pnode.append(n)
            zs.append((p - mu) @ W)
        node_z[n] = np.mean(zs, 0)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zq = (T - mu) @ W
    Zp = (P - mu) @ W
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
    log(f"top-1 路由精度 {float(np.mean(top_cls == yT)):.4f}（三档相同）")

    NZ = np.stack([node_z[n] for n in range(Nn)])
    dnn = np.sqrt(((NZ[:, None, :] - NZ[None, :, :]) ** 2).sum(-1))
    nbr8 = {n: list(np.argsort(dnn[n])[:8]) for n in range(Nn)}

    Hn, yn = {}, {}
    for n in range(Nn):
        m = tr_node == n
        Hn[n] = Htr[m]
        yn[n] = np.array([nc[n].index(c) for c in ytr[m]], dtype=np.int64)

    res = {"n_nodes": Nn, "d": int(d_eff),
           "route_top1": float(np.mean(top_cls == yT)),
           "body_kb": (384 * 256 * 4 + 256 * 2 * 4) / 1024}

    def evaluate(tag, bodies, heads):
        right = 0
        s = []
        for n in range(Nn):
            lg = infer(bodies[n], heads[n], T)
            pred = np.array(nc[n])[lg.argmax(1)]
            m = top_node == n
            if m.sum():
                right += int((pred[m] == yT[m]).sum())
            s.append(float(np.mean(pred == yT)))
        res[tag] = dict(D1=right / len(T), S_global_mean=float(np.mean(s)))
        log(f"{tag:26s} D1 {res[tag]['D1']:.4f}   各节点全局精度均值 {res[tag]['S_global_mean']:.4f}")

    t0 = time.time()
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(None, 2)
        train_join(Hn[n], yn[n], bodies[n], heads[n], 300)
    evaluate("E0 无 body", bodies, heads)
    log(f"  用时 {time.time() - t0:.0f}s")

    t0 = time.time()
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(256, 2)
        train_join(Hn[n], yn[n], bodies[n], heads[n], 300)
    evaluate("E1 body 本地训练", bodies, heads)
    log(f"  用时 {time.time() - t0:.0f}s")

    if os.environ.get("NO_SYNC") != "1":
        t0 = time.time()
        bodies, heads = {}, {}
        init = None
        for n in range(Nn):
            bodies[n], heads[n] = build(256, 2)
            if init is None:
                init = {k: v.clone() for k, v in bodies[n].state_dict().items()}
            bodies[n].load_state_dict(init)
        for r in range(3):
            for n in range(Nn):
                train_join(Hn[n], yn[n], bodies[n], heads[n], 150)
            for n in range(Nn):
                sd = bodies[n].state_dict()
                avg = {k: torch.stack([bodies[j].state_dict()[k] for j in nbr8[n]]).mean(0)
                       for k in sd}
                bodies[n].load_state_dict(avg)
            for n in range(Nn):
                heads[n] = build(256, 2)[1]
                for p in bodies[n].parameters():
                    p.requires_grad_(False)
                opt = torch.optim.Adam(heads[n].parameters(), lr=5e-3, weight_decay=1e-4)
                with torch.no_grad():
                    F = bodies[n](torch.from_numpy(Hn[n]))
                yt = torch.from_numpy(yn[n])
                for _ in range(300):
                    opt.zero_grad()
                    nn.CrossEntropyLoss()(heads[n](F), yt).backward()
                    opt.step()
                for p in bodies[n].parameters():
                    p.requires_grad_(True)
            log(f"   E2 第 {r + 1} 轮完成")
        evaluate("E2 body 近邻同步", bodies, heads)
        log(f"  用时 {time.time() - t0:.0f}s")

    with open(os.path.join(ROOT, "results", "results_body_head_c100.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_body_head_c100.json")


if __name__ == "__main__":
    main()
