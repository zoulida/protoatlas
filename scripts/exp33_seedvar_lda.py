"""exp33: 端到端读数的随机种子波动（判别投影口径）

目的：正文 5.3 节称各方案差值「落在五个随机种子的波动范围之内」，
需要有实测的波动幅度。本脚本把 exp32 的流程在 5 个全局种子下重跑，
只报集中式、FedAvg 与本文单原型／8 子原型四者的端到端精度，
以及彼此的极差，用于判断 0.02 至 0.4 个百分点这一量级的差值是否可信。

随机性来源：线性分类器的初始化与优化顺序。数据划分、判别投影、
k-means 拆簇均固定（k-means 以类号为种子），故波动只来自训练侧。
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp32_end2end_all_lda import fit_lda, train_linear, predict, kmeans  # noqa: E402


def log(*a):
    print(*a, flush=True)


def run_once(H, T, y_tr, y_te, tr_node, nc, Nn, cls_of_node, seed):
    """一次完整流程；seed 只作用于训练侧"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # A 集中式
    mA = train_linear(H, y_tr, 10, seed=seed)
    accA = float(np.mean(predict(mA, T).argmax(1) == y_te))

    # B 各节点本地 head（预测结果供 D 复用）
    heads = {}
    for n in range(Nn):
        m = tr_node == n
        lab = np.array([cls_of_node[n].index(c) for c in y_tr[m]])
        heads[n] = train_linear(H[m], lab, len(cls_of_node[n]), seed=seed * 100 + n)
    pred_all = {n: np.array(cls_of_node[n])[predict(heads[n], T).argmax(1)]
                for n in range(Nn)}

    # C FedAvg
    torch.manual_seed(seed * 1000 + 7)
    glob = nn.Linear(384, 10)
    gsd = glob.state_dict()
    for _ in range(20):
        states = []
        for n in range(Nn):
            m = tr_node == n
            Xn, yn = torch.from_numpy(H[m]), torch.from_numpy(y_tr[m])
            loc = nn.Linear(384, 10)
            loc.load_state_dict(gsd)
            opt = torch.optim.SGD(loc.parameters(), lr=0.05, momentum=0.9)
            lossf = nn.CrossEntropyLoss()
            for _ in range(3):
                opt.zero_grad()
                lossf(loc(Xn), yn).backward()
                opt.step()
            states.append({k: v.clone() for k, v in loc.state_dict().items()})
        gsd = {k: torch.stack([s[k] for s in states]).mean(0) for k in gsd}
    glob.load_state_dict(gsd)
    accC = float(np.mean(predict(glob, T).argmax(1) == y_te))

    # D 判别投影 + 单原型 / 8 子原型
    mu, W, d_eff = fit_lda(H, y_tr, 9)
    Zq = (T - mu) @ W
    out = {"A": accA, "C": accC}
    for k in (1, 8):
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in nc[n]:
                rows = (tr_node == n) & (y_tr == c)
                X = H[rows]
                if len(X) < k:
                    continue
                Cs = X.mean(0, keepdims=True) if k == 1 else kmeans(X, k, seed=c)
                for v in Cs:
                    protos.append(v)
                    pcls.append(c)
                    pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zp = (P - mu) @ W
        D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                               - 2 * Zq @ Zp.T, 0))
        o = np.argsort(D, 1)
        pred = np.zeros(len(T), np.int32)
        hit1 = 0
        for i in range(len(T)):
            seen, pick = set(), []
            for j in o[i]:
                c = int(pcls[j])
                if c not in seen:
                    seen.add(c)
                    pick.append(j)
                if len(pick) == 3:
                    break
            top = int(pick[0])
            hit1 += int(pcls[top] == y_te[i])
            pred[i] = pred_all[int(pnode[top])][i]
        out[f"D{k}"] = float(np.mean(pred == y_te))
        out[f"hit1_{k}"] = hit1 / len(T)
    return out


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    rows = []
    for s in range(5):
        r = run_once(H, T, y_tr, y_te, tr_node, nc, Nn, cls_of_node, seed=s + 1)
        rows.append(r)
        log(f"种子 {s + 1}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))

    keys = ["A", "C", "D1", "D8"]
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        summ[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                       lo=float(v.min()), hi=float(v.max()),
                       range=float(v.max() - v.min()))
    log("")
    for k in keys:
        s = summ[k]
        log(f"{k}: 均值 {s['mean']:.4f}  标准差 {s['std']:.4f}  "
            f"极差 {s['range']:.4f}（{s['range'] * 100:.2f} 个百分点）")

    for a, b in [("A", "D1"), ("C", "D1"), ("A", "D8"), ("C", "D8")]:
        da = np.array([r[a] for r in rows])
        db = np.array([r[b] for r in rows])
        log(f"{b} − {a}: 均值 {np.mean(db - da) * 100:+.2f} 个百分点，"
            f"逐种子 {', '.join(f'{(x - y) * 100:+.2f}' for x, y in zip(db, da))}")

    with open(os.path.join(ROOT, "results", "results_seedvar_lda.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(rows=rows, summary=summ), f, ensure_ascii=False, indent=2)
    log("写入 results_seedvar_lda.json")


if __name__ == "__main__":
    main()
