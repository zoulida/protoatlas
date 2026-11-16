"""exp34: 端到端全档位的五种子重复（判别投影口径）

背景：exp32 单次运行的读数为 A=0.9646、C=0.9684、D1=0.9682，
正文据此写「本文与 FedAvg 持平」。exp33 用五个种子重复后发现
C（FedAvg）的种子波动达 0.28 个百分点，单次的 0.9684 属偏高值，
而五次均值为 0.9645，低于本文的 0.9680。结论方向因此与单次相反。

本脚本把 A/B/C 与 k=1/2/3/5/8 全部档位在 5 个种子下重跑，
输出均值与样本标准差，作为表 5-1 与表 5-2 的正式读数。

口径：坐标一律取判别投影，d=K-1=9；数据划分、投影、k-means 固定，
波动只来自线性分类器的初始化与优化顺序。
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

KS = (1, 2, 3, 5, 8)
SEEDS = (1, 2, 3, 4, 5)


def log(*a):
    print(*a, flush=True)


def build_vnodes(H, tr_node, nc, Nn, k):
    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        for c in nc[n]:
            rows = (tr_node == n) & (y_tr_global == c)
            X = H[rows]
            if len(X) < k:
                continue
            Cs = X.mean(0, keepdims=True) if k == 1 else kmeans(X, k, seed=c)
            for v in Cs:
                protos.append(v)
                pcls.append(c)
                pnode.append(n)
    return np.stack(protos).astype(np.float32), np.array(pcls), np.array(pnode)


def route(Zq, Zp, pcls, pnode, pred_all, y_te):
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2 * Zq @ Zp.T, 0))
    o = np.argsort(D, 1)
    r3, h1 = 0, 0
    pred = np.zeros(len(Zq), np.int32)
    for i in range(len(Zq)):
        seen, pick = set(), []
        for j in o[i]:
            c = int(pcls[j])
            if c not in seen:
                seen.add(c)
                pick.append(j)
            if len(pick) == 3:
                break
        r3 += int(y_te[i] in [int(pcls[j]) for j in pick])
        top = int(pick[0])
        h1 += int(pcls[top] == y_te[i])
        pred[i] = pred_all[int(pnode[top])][i]
    n = len(Zq)
    return r3 / n, h1 / n, float(np.mean(pred == y_te))


def ridge_probe(H, y, n_cls, lam=1.0):
    """闭式 ridge 线性分类器：同一族里的最强解，作为集中式参照"""
    X = H.astype(np.float64)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.zeros((len(X), n_cls), np.float64)
    Y[np.arange(len(X)), y] = 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def main():
    global y_tr_global
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    y_tr_global = y_tr
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    # 集中式参照之一：闭式 ridge（无随机性，与种子无关）
    Wr = ridge_probe(H, y_tr, 10)
    Tr = np.hstack([T.astype(np.float64), np.ones((len(T), 1))])
    accR = float(np.mean((Tr @ Wr).argmax(1) == y_te))
    log(f"集中式 ridge（闭式）                        {accR:.4f}")

    rec = {k: [] for k in ["A", "B", "C"] + [f"k{k}" for k in KS]}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        mA = train_linear(H, y_tr, 10, seed=seed)
        accA = float(np.mean(predict(mA, T).argmax(1) == y_te))

        heads = {}
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([cls_of_node[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(cls_of_node[n]), seed=seed * 100 + n)
        pred_all = {n: np.array(cls_of_node[n])[predict(heads[n], T).argmax(1)]
                    for n in range(Nn)}
        accB = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))

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

        mu, W, d_eff = fit_lda(H, y_tr, 9)
        Zq = (T - mu) @ W

        line = f"种子 {seed}: A={accA:.4f} B={accB:.4f} C={accC:.4f}"
        rec["A"].append(accA)
        rec["B"].append(accB)
        rec["C"].append(accC)
        for k in KS:
            P, pcls, pnode = build_vnodes(H, tr_node, nc, Nn, k)
            Zp = (P - mu) @ W
            r3, h1, e2e = route(Zq, Zp, pcls, pnode, pred_all, y_te)
            rec[f"k{k}"].append(dict(n_vnode=len(P), recall3=r3, hit1=h1, end2end=e2e))
            line += f"  k{k}:{r3:.4f}/{h1:.4f}/{e2e:.4f}"
        log(line)

    out = {"seeds": list(SEEDS), "d": 9, "proj": "lda", "ridge": accR,
           "acc": {}, "sub": {}}
    log("")
    for k in ("A", "B", "C"):
        v = np.array(rec[k])
        out["acc"][k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                             lo=float(v.min()), hi=float(v.max()))
        log(f"{k}: {v.mean():.4f} ± {v.std(ddof=1):.4f}"
            f"  极差 {v.max() - v.min():.4f}")
    for k in KS:
        row = {}
        for f in ("n_vnode", "recall3", "hit1", "end2end"):
            a = np.array([r[f] for r in rec[f"k{k}"]], dtype=float)
            row[f] = dict(mean=float(a.mean()), std=float(a.std(ddof=1)))
        out["sub"][str(k)] = row
        log(f"k={k}: vnode {int(row['n_vnode']['mean'])}  "
            f"召回@3 {row['recall3']['mean']:.4f}  "
            f"命中@1 {row['hit1']['mean']:.4f}±{row['hit1']['std']:.4f}  "
            f"端到端 {row['end2end']['mean']:.4f}±{row['end2end']['std']:.4f}")

    a = np.array(rec["A"])
    c = np.array(rec["C"])
    for name, arr in (("A 同配方集中式", a), ("C FedAvg", c), ("ridge 集中式", accR)):
        for k in KS:
            b = np.array([r["end2end"] for r in rec[f"k{k}"]])
            dif = (b - arr) * 100
            log(f"k={k} − {name}: 均值 {dif.mean():+.2f} 点，逐种子 "
                + ", ".join(f"{x:+.2f}" for x in dif))

    with open(os.path.join(ROOT, "results", "results_end2end_seedavg_lda.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("写入 results_end2end_seedavg_lda.json")


if __name__ == "__main__":
    main()
