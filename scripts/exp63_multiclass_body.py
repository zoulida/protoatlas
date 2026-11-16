"""exp63: 本地判别难度上升后，body 是否开始有用

背景：exp61/62 在「每节点 2 类」下放开 body，端到端精度没有变化——本地判别
早已饱和（候选内本地答对 0.9952）。作者提出「1000 类才看得出效果」，但千类
档只有 ImageNet-1k 验证集、每类 50 张（30 张作原型池），撑不起多副本的节点
划分。本脚本改用另一条路把本地判别推离饱和：加大每节点持有的类数。

CIFAR-100，每类 500 张训练、测试 3000 张，100 类均匀分给 Nn 个节点：
  Nn=20 → 每节点 5 类；Nn=10 → 每节点 10 类；Nn=5 → 每节点 20 类
坐标仍由协议资产 φ 与判别投影决定，body 不进入路由，故候选列表与路由读数
在各档、各配置下完全相同。每档比较两档判别器：
  E0  body 取恒等，head 直接接 384 维
  E1  body = 384→256→ReLU，与 head 联合训练
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
from exp53_c100_end2end_lda import load_c100  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
PER_CLASS_TRAIN = 300
N_TEST = 3000


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


def run(H, y, T, yT, Nn):
    K = 100
    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(K), Nn))}
    log(f"\n=== {Nn} 节点，每节点 {K // Nn} 类 ===")

    mu, W, d_eff = fit_lda(H, y, K - 1)
    log(f"判别投影 d={d_eff}")

    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        for c in cls_of_node[n]:
            X = H[y == c]
            if len(X) == 0:
                continue
            protos.append(X.mean(0))
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
    log(f"top-1 路由精度 {route:.4f}")

    Hn, yn = {}, {}
    for n in range(Nn):
        rows = np.isin(y, cls_of_node[n])
        idx = np.where(rows)[0]
        Hn[n] = H[idx]
        yn[n] = np.array([cls_of_node[n].index(c) for c in y[idx]], dtype=np.int64)

    out = {"n_nodes": Nn, "classes_per_node": K // Nn, "route_top1": route}

    def evaluate(tag, bodies, heads):
        right = 0
        for n in range(Nn):
            pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
            m = top_node == n
            if m.sum():
                right += int((pred[m] == yT[m]).sum())
        out[tag] = right / len(T)
        log(f"  {tag:16s} D1 {out[tag]:.4f}")

    for tag, hid in (("E0 无 body", None), ("E1 body 本地", 256)):
        t0 = time.time()
        bodies, heads = {}, {}
        for n in range(Nn):
            bodies[n], heads[n] = build(hid, K // Nn)
            train_join(Hn[n], yn[n], bodies[n], heads[n], 300)
        evaluate(tag, bodies, heads)
        log(f"     用时 {time.time() - t0:.0f}s")
    return out


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    keep = np.isin(y, np.arange(100))
    H, y = H[keep], y[keep]
    res = {}
    for Nn in (20, 10, 5):
        res[str(Nn)] = run(H, y, T, yT, Nn)
    with open(os.path.join(ROOT, "results", "results_multiclass_body.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_multiclass_body.json")


if __name__ == "__main__":
    main()
