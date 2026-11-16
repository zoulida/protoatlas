"""exp64: body 的增益是否取决于每节点样本量

起因：作者判断 exp61/62/63 里 body 增益低是「样本太少」所致，而不是本地
判别饱和。本脚本把样本量单独拆出来当自变量。

设定：CIFAR-100，5 节点 × 20 类（exp63 中增益最大的一档）。路由与候选
完全不动（坐标只由协议资产 φ 决定，body 不进路由），只改每个节点训练
判别器时实际用到的样本数：

  每类 shots 张 → 每节点 20×shots 张
  shots = 25 / 50 / 100 / 250 / 500（CIFAR-100 每类上限 500）

每档比两配置：
  E0  body 取恒等，head 直接接 384 维
  E1  body = 384→256→ReLU，与 head 联合训练

若增益随 shots 单调上升，则「样本不足」成立；若在 shots=500（每节点
10 000 张，远多于 384→256 的 9.8 万参数所需）仍不足一个点，则样本量
不是唯一原因。
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

N_NODES = 5             # 100 类 / 5 = 每节点 20 类
N_TEST = 3000
SHOTS = (25, 50, 100, 250, 500)


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
    K = 100
    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(K), N_NODES))}
    log(f"数据：训练 {H.shape} 测试 {T.shape}，{N_NODES} 节点 × {K // N_NODES} 类")

    # ---------- 路由：与 exp63 完全相同，各档共用
    mu, W, d_eff = fit_lda(H, y, K - 1)
    log(f"判别投影 d={d_eff}")
    protos, pcls, pnode = [], [], []
    for n in range(N_NODES):
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
    log(f"top-1 路由精度 {route:.4f}（各档共用）")

    rows_of = {}
    for n in range(N_NODES):
        rows_of[n] = {c: np.where(y == c)[0] for c in cls_of_node[n]}

    res = {}
    for shots in SHOTS:
        rng = np.random.RandomState(7)
        Hn, yn = {}, {}
        for n in range(N_NODES):
            idxs = []
            for c in cls_of_node[n]:
                sel = rows_of[n][c]
                take = rng.choice(sel, size=min(shots, len(sel)), replace=False)
                idxs.append(take)
            idx = np.concatenate(idxs)
            Hn[n] = H[idx]
            yn[n] = np.array([cls_of_node[n].index(c) for c in y[idx]], dtype=np.int64)

        out = {"shots_per_class": shots,
               "samples_per_node": int(sum(len(v) for v in yn.values()) / N_NODES),
               "route_top1": route}
        log(f"\n=== 每类 {shots} 张，每节点 {out['samples_per_node']} 张 ===")

        def evaluate(bodies, heads):
            right = 0
            for n in range(N_NODES):
                pred = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
                m = top_node == n
                if m.sum():
                    right += int((pred[m] == yT[m]).sum())
            return right / len(T)

        for tag, hid in (("E0", None), ("E1", 256)):
            t0 = time.time()
            bodies, heads = {}, {}
            for n in range(N_NODES):
                bodies[n], heads[n] = build(hid, K // N_NODES)
                train_join(Hn[n], yn[n], bodies[n], heads[n], 300)
            out[tag] = evaluate(bodies, heads)
            log(f"  {tag} D1 {out[tag]:.4f}   用时 {time.time() - t0:.0f}s")
        out["gain_pt"] = round((out["E1"] - out["E0"]) * 100, 2)
        log(f"  增益 {out['gain_pt']:+.2f} 点")
        res[str(shots)] = out

    with open(os.path.join(ROOT, "results", "results_shots_body.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n写入 results_shots_body.json")


if __name__ == "__main__":
    main()
