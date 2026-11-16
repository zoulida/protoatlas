"""exp14: 拒识到底靠什么？
对比两种判据：
  A 几何判据  —— 到本节点原型的距离 <= 本地 p95 半径
  B 判别判据  —— 本地 head 的 margin（top1 logit - top2 logit）

核心指标：错送过来的样本，节点能否拒识；同时不误拒自己的样本。
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


def log(*a):
    print(*a, flush=True)


def d2mat(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def auroc(score_in, score_out):
    """score 越高越像 in-domain。返回 AUROC（0.5=瞎猜）。"""
    s = np.concatenate([score_in, score_out])
    y = np.concatenate([np.ones(len(score_in)), np.zeros(len(score_out))])
    order = np.argsort(s)
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_head(X, y, n_cls=2, steps=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.int64))
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(m(xt), yt)
        loss.backward()
        opt.step()
    return m


def main():
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]
    NAME = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            if rows.sum() == 0:
                continue
            protos.append(H[rows].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zq, Zp = (T - mu) @ W, (P - mu) @ W
    D = np.sqrt(d2mat(Zq, Zp))
    order = np.argsort(D, axis=1)

    # ---------- 训练 20 个本地 head，并记录 margin 分布 ----------
    log("训练 20 个节点本地 head（384→2 线性，冻结 DINOv2 特征）...")
    heads, geo_r, id_margin, ood_margin = {}, {}, {}, {}
    for n in range(C["n_nodes"]):
        m = tr_node == n
        cls = sorted(nc[n])
        Xn = H[m]
        yn = np.array([cls.index(c) for c in y_tr[m]])
        heads[n] = train_head(Xn, yn)
        # 几何半径
        zl = (Xn - mu[0]) @ W
        for c in cls:
            rows = y_tr[m] == c
            pr = zl[rows].mean(0)
            dloc = np.sqrt((((zl[rows] - pr)) ** 2).sum(1))
            geo_r[(n, c)] = float(np.percentile(dloc, 95))
        # margin / energy 分布：本地验证（用测试集里属于本节点的样本）
        mi = [i for i in range(len(y_te)) if y_te[i] in cls]
        mo = [i for i in range(len(y_te)) if y_te[i] not in cls]
        with torch.no_grad():
            lg_i = heads[n](torch.from_numpy(T[mi])).numpy()
            lg_o = heads[n](torch.from_numpy(T[mo])).numpy()
        id_margin[n] = lg_i.max(1) - np.partition(lg_i, -2, axis=1)[:, -2]
        ood_margin[n] = lg_o.max(1) - np.partition(lg_o, -2, axis=1)[:, -2]
        assert len(mi) + len(mo) == len(y_te)

    log("\n" + "=" * 78)
    log("① 两种判据的区分能力（能不能把'别人的东西'认出来）")
    log("=" * 78)
    all_id = np.concatenate([id_margin[n] for n in range(C["n_nodes"])])
    all_ood = np.concatenate([ood_margin[n] for n in range(C["n_nodes"])])
    log(f"  margin 判据：ID 中位 {np.median(all_id):.2f}，OOD 中位 {np.median(all_ood):.2f}，"
        f"AUROC = {auroc(all_id, all_ood):.4f}")

    # 几何判据的 AUROC：用"到最近原型距离 / 本节点中位半径"作分数（取负）
    gscore_id, gscore_ood = [], []
    for n in range(C["n_nodes"]):
        cls = sorted(nc[n])
        idx = [j for j in range(len(P)) if pnode[j] == n]
        med = np.median([np.median([geo_r.get((n, k), 1.0)]) for k in cls])
        for i in range(len(y_te)):
            dmin = D[i, idx].min() / max(med, 1e-9)
            (gscore_id if y_te[i] in cls else gscore_ood).append(-dmin)
    log(f"  几何判据：AUROC = {auroc(np.array(gscore_id), np.array(gscore_ood)):.4f}")

    # ---------- ② 端到端 ----------
    log("\n" + "=" * 78)
    log("② 端到端：按类去重 top-3 依次询问，第一个受理的答")
    log("=" * 78)

    def run(mode, q):
        ok = wrong = rej = 0
        blocked = 0     # 第一跳送错且被拒识
        for i in range(len(y_te)):
            z = Zq[i]
            seen, cand = set(), []
            for j in order[i]:
                c = pcls[j]
                if c not in seen:
                    seen.add(c)
                    cand.append((c, pnode[j]))
                if len(cand) == 3:
                    break
            got = None
            for pos, (c, n) in enumerate(cand):
                if mode == "geo":
                    j = next(jj for jj in range(len(P)) if pnode[jj] == n and pcls[jj] == c)
                    d = float(np.sqrt(((z - Zp[j]) ** 2).sum()))
                    accept = d <= geo_r[(n, c)]
                else:
                    cls = sorted(nc[n])
                    with torch.no_grad():
                        lg = heads[n](torch.from_numpy(T[i][None, :])).numpy()[0]
                    mg = lg.max() - np.partition(lg, -2)[-2]
                    # q = 希望放行的比例；margin 越高越像自己，故取 100-q 分位作阈值
                    thr = np.percentile(id_margin[n], 100 - q)
                    accept = mg >= thr
                if pos == 0 and (not accept):
                    if c != y_te[i]:
                        blocked += 1
                if accept:
                    got = c
                    break
            if got is None:
                rej += 1
            elif got == y_te[i]:
                ok += 1
            else:
                wrong += 1
        N = len(y_te)
        return dict(correct=ok / N, dangerous=wrong / N, all_rejected=rej / N,
                    wrong_blocked=blocked / N)

    res = {}
    for mode, qs in [("geo", [95]), ("head", [90, 95, 99])]:
        for q in qs:
            r = run(mode, q)
            key = f"{mode}-p{q}"
            res[key] = r
            log(f"\n[{key}]")
            log(f"  答对                    {r['correct']:.4f}")
            log(f"  危险错误（受理且答错）  {r['dangerous']:.4f}")
            log(f"  全部拒识                {r['all_rejected']:.4f}")
            log(f"  └ 第一跳送错被挡住       {r['wrong_blocked']:.4f}")

    # 拒识后回退时的"覆盖-错误"权衡
    log("\n" + "=" * 78)
    log("③ margin 阈值的权衡曲线（head 判据）")
    log("=" * 78)
    log(f"  {'分位':<8}{'答对':<10}{'危险错误':<12}{'全部拒识':<10}")
    for q in [70, 80, 90, 95, 99]:
        r = run("head", q)
        log(f"  p{q:<7}{r['correct']:<10.4f}{r['dangerous']:<12.4f}{r['all_rejected']:<10.4f}")

    with open(os.path.join(ROOT, "results", "results_reject2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
