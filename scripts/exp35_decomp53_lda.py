"""exp35: 5.3 设定下的三项分解实测（A、S_exp、eps、E2E）

目的：附录 C.2 表 C-2 的 CIFAR-10 行为 G=0.9692、A=0.9674、S_exp=0.9906、
E2E=0.9583，其中 S_exp 在实验目录里查不到生成脚本；而第 5.3 节按 LDA d=9、
按类去重的分布式口径测得命中@1 为 0.9682、端到端为 0.9680，两者相差 0.02 点，
意味着 S_exp 接近 1。本脚本在 5.3 的同一划分、同一投影、同一 head 上直接统计
A、S_exp、eps，用一次运行把两者的关系定下来。

统计口径（逐查询）：
  路由是否正确 ro = (终点 vnode 的类 == 真类)
  终判是否正确 ok = (接收 head 给出的类 == 真类)
  A     = mean(ro)
  S_exp = mean(ok | ro)
  eps   = mean(ok | not ro)
  E2E   = mean(ok)，并校验 E2E == A*S_exp + (1-A)*eps
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

from exp32_end2end_all_lda import fit_lda, train_linear, predict  # noqa: E402

SEEDS = (1, 2, 3, 4, 5)


def log(*a):
    print(*a, flush=True)


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        heads = {}
        for n in range(Nn):
            m = tr_node == n
            lab = np.array([cls_of_node[n].index(c) for c in y_tr[m]])
            heads[n] = train_linear(H[m], lab, len(cls_of_node[n]), seed=seed * 100 + n)
        pred_all = {n: np.array(cls_of_node[n])[predict(heads[n], T).argmax(1)]
                    for n in range(Nn)}

        # 单原型 vnode：每节点每类的样本均值
        protos, pcls, pnode = [], [], []
        for n in range(Nn):
            for c in nc[n]:
                rows_m = (tr_node == n) & (y_tr == c)
                protos.append(H[rows_m].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)

        mu, W, d_eff = fit_lda(H, y_tr, 9)
        Zq, Zp = (T - mu) @ W, (P - mu) @ W
        D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                               - 2 * Zq @ Zp.T, 0))
        top = np.argmin(D, 1)                      # 全局最近 vnode（单点，无路由过程）
        ro = pcls[top] == y_te                     # 路由是否正确
        got = np.array([pred_all[int(pnode[t])][i] for i, t in enumerate(top)])
        ok = got == y_te

        A = float(ro.mean())
        S = float(ok[ro].mean())
        eps = float(ok[~ro].mean())
        E = float(ok.mean())
        chk = A * S + (1 - A) * eps
        log(f"种子 {seed}: A={A:.4f}  S_exp={S:.4f}  eps={eps:.4f}  "
            f"E2E={E:.4f}  A*S+(1-A)*eps={chk:.4f}  跳转类别数={len(P)}")
        rows.append(dict(seed=seed, A=A, S=S, eps=eps, E2E=E, chk=chk,
                         n_wrong=int((~ro).sum()), n_wrong_ok=int(ok[~ro].sum())))

    a = np.array([r["A"] for r in rows])
    s = np.array([r["S"] for r in rows])
    e = np.array([r["eps"] for r in rows])
    q = np.array([r["E2E"] for r in rows])
    log("")
    log(f"A     = {a.mean():.4f} ± {a.std(ddof=1):.4f}")
    log(f"S_exp = {s.mean():.4f} ± {s.std(ddof=1):.4f}")
    log(f"eps   = {e.mean():.4f} ± {e.std(ddof=1):.4f}")
    log(f"E2E   = {q.mean():.4f} ± {q.std(ddof=1):.4f}")
    log(f"路由错误样本约 {np.mean([r['n_wrong'] for r in rows]):.0f} 条，"
        f"其中终判对 {np.mean([r['n_wrong_ok'] for r in rows]):.0f} 条")

    with open(os.path.join(ROOT, "results", "results_decomp53_lda.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(summary=dict(A=float(a.mean()), S=float(s.mean()),
                                    eps=float(e.mean()), E2E=float(q.mean())),
                       rows=rows), f, ensure_ascii=False, indent=2)
    log("写入 results_decomp53_lda.json")


if __name__ == "__main__":
    main()
