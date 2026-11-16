"""exp55: 路由完美的上界（oracle routing）

问题：表 5-1 里本文与集中式的差额全在路由一侧，若把路由做到全对，端到端能到
多少？会不会超过集中式？

做法：对每个测试样本，直接把请求送到一个持有其真类的节点（绕过路由），由该
节点的本地 head 作答，算准确率。这个读数记为 S_oracle，即 A=1 时的端到端。

同时报三个分解量：
  A          当前路由的命中@1
  S_exp      当前路由正确的子集上本地 head 判对的条件概率（与 exp34/exp53 同口径）
  S_on_wrong 当前路由错误的子集上，若改为送对节点，本地 head 判对的比例
             —— 它说明"回收路由损失"能拿回多少，也检验 S_exp 是否因样本选择
                偏向易分样本而偏高

两个数据集：CIFAR-10（20 节点 × 2 类，d=9）与 CIFAR-100（200 节点 × 2 类，d=99），
设定与 exp34 / exp53 逐处一致，五随机种子。
"""
import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
import sys
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from exp32_end2end_all_lda import fit_lda, train_linear, predict  # noqa: E402
from exp1_route import make_shard2  # noqa: E402

SEEDS = (1,) if os.environ.get("ONE_SEED") == "1" else (1, 2, 3, 4, 5)
OUT = os.path.join(ROOT, "results", "results_oracle_route.json")


def log(*a):
    print(*a, flush=True)


def owners_table(cls_of_node, n_cls):
    """每个类被哪些节点持有"""
    return {c: [n for n in range(len(cls_of_node)) if c in cls_of_node[n]]
            for c in range(n_cls)}


def oracle_stats(pred_all, owners, y_te, hit1_mask):
    """pred_all: {node: 该节点对全部测试样本的预测}

    返回 (S_oracle_first, S_on_wrong, S_on_right, ok_any, ok_major)
    first  = 取持有真类的节点中序号最小的一个作答
    major  = 取全部副本的多数票（副本数相同时取 first 那一票）
    any    = 至少一个副本判对（乐观上界，真实系统不会用到）
    """
    n = len(y_te)
    ok_first = np.zeros(n, np.int8)
    ok_major = np.zeros(n, np.int8)
    ok_any = np.zeros(n, np.int8)
    for i in range(n):
        c = int(y_te[i])
        ns = owners[c]
        votes = np.array([int(pred_all[m][i] == c) for m in ns])
        ok_first[i] = votes[0]
        ok_major[i] = int(votes.sum() * 2 > len(votes)) or (
            votes.sum() * 2 == len(votes) and votes[0] == 1)
        ok_any[i] = int(votes.any())
    right = hit1_mask.astype(bool)
    wrong = ~right
    s_oracle = float(ok_first.mean())
    s_on_right = float(ok_first[right].mean()) if right.any() else float("nan")
    s_on_wrong = float(ok_first[wrong].mean()) if wrong.any() else float("nan")
    return s_oracle, s_on_wrong, s_on_right, float(ok_any.mean()), float(ok_major.mean())


def run_case(name, H, y_tr, T, y_te, nc, tr_node, K, d):
    Nn = len(nc)
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    owners = owners_table(cls_of_node, K)
    log(f"\n===== {name}: K={K} 节点={Nn} d={d} 训练 {len(y_tr)} 测试 {len(y_te)}")
    log(f"每类副本数 {sorted(set(len(v) for v in owners.values()))}")

    mu, W, d_eff = fit_lda(H, y_tr, d)
    Zq = (T - mu) @ W
    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        for c in nc[n]:
            rows = (tr_node == n) & (y_tr == c)
            if rows.sum() == 0:
                continue
            protos.append(H[rows].mean(0))
            pcls.append(c)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zp = (P - mu) @ W
    # 当前路由：命中@1 掩码
    D2 = ((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2.0 * Zq @ Zp.T)
    nn_idx = np.argmin(D2, 1)
    hit1_mask = (pcls[nn_idx] == y_te)

    rec = {k: [] for k in ("s_oracle", "s_on_wrong", "s_on_right",
                           "ok_any", "ok_major", "e2e")}
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
        # 当前路由的端到端
        e2e = float(np.mean([pred_all[int(pnode[j])][i] == y_te[i]
                             for i, j in enumerate(nn_idx)]))
        s_o, s_w, s_r, ok_any, ok_major = oracle_stats(pred_all, owners, y_te, hit1_mask)
        for k, v in (("s_oracle", s_o), ("s_on_wrong", s_w), ("s_on_right", s_r),
                     ("ok_any", ok_any), ("ok_major", ok_major), ("e2e", e2e)):
            rec[k].append(v)
        log(f"种子 {seed}: A={hit1_mask.mean():.4f}  E2E={e2e:.4f}  "
            f"S_oracle={s_o:.4f}  S_on_wrong={s_w:.4f}  "
            f"多数票={ok_major:.4f}  任一={ok_any:.4f}")
        del heads, pred_all

    out = {"K": K, "n_nodes": Nn, "d": d, "n_test": int(len(y_te)),
           "A": float(hit1_mask.mean()), "n_vnode": int(len(P))}
    for k in rec:
        v = np.array(rec[k], dtype=float)
        out[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)))
        log(f"{k:12s} {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    return out


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    res = {}

    # ---------- CIFAR-10 ----------
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    nc = [sorted(x) for x in nc]
    res["c10"] = run_case("CIFAR-10", H, y_tr, T, y_te, nc, tr_node, 10, 9)
    del H, T, y_tr, y_te, nc, tr_node, tr_sel, te_sel

    # ---------- CIFAR-100 ----------
    import pandas as pd
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    dd = np.load(os.path.join(CACHE, "c100_train.npz"))
    y_all = dd["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte100 = te["fine_label"].to_numpy().astype(np.int64)
    n_te = min(len(Hte), len(yte100))
    Hte, yte100 = Hte[:n_te], yte100[:n_te]
    n_tr = min(len(Htr), len(y_all))
    Htr, y_all = Htr[:n_tr], y_all[:n_tr]

    K, Nn, PER = 100, 200, 2
    nc100, ni100 = make_shard2(y_all, Nn, K, PER, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni100):
        for c in nc100[n]:
            sel = idx[y_all[idx] == c]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    res["c100"] = run_case("CIFAR-100", Htr[tr_sel], y_all[tr_sel], Hte, yte100,
                           [sorted(x) for x in nc100], tr_node, K, K - 1)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"\n写入 {os.path.basename(OUT)}")

    log("\n===== 结论对照 =====")
    G = {"c10": (0.9692, 0.9656), "c100": (0.8230, 0.8308)}
    for key, label in (("c10", "CIFAR-10"), ("c100", "CIFAR-100")):
        r = res[key]
        g_open, g_grad = G[key]
        log(f"{label}: A={r['A']:.4f} E2E={r['e2e']['mean']:.4f} "
            f"S_oracle={r['s_oracle']['mean']:.4f}")
        log(f"  集中式 闭式 {g_open:.4f} / 同头梯度 {g_grad:.4f}")
        log(f"  路由完美时 E2E − 集中式(闭式) = {(r['s_oracle']['mean']-g_open)*100:+.2f} 点")
        log(f"  路由完美时 E2E − 集中式(梯度) = {(r['s_oracle']['mean']-g_grad)*100:+.2f} 点")


if __name__ == "__main__":
    main()
