"""exp52: 大规模覆盖网（K=10^3, R=35, M=35000）的路由实验

对应正文 6.6 的规模档与 4.3.4 的多起点/一跳扩散。
与表 6-9 的差别只在两点：类数由 10^2 提到 10^3，副本因子固定为 35。

口径：
  · 投影取判别投影（LDA），d = min(K-1, D)，K>385 时封顶 384（全维白化）
  · 近邻表长度 k = round(max(8, 6 + 4*log10(M/100)))
  · 起点取 8 个随机 vnode 中距查询最近的 1 个（与 6.2 同口径）
  · 3 起点取最近 3 个互不同类的起点并行下降
  · 1 跳扩散在单起点终点上取终点与其邻居合并排序

用法：
  python exp52_scale_route.py --data in1k --R 35
  python exp52_scale_route.py --data c100 --R 35
"""
import os
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
ELL = 8  # 标识字节数，与正文表 4-2 一致


def log(*a):
    print(*a, flush=True)


def sqdist(A, B):
    """欧氏距离的平方，A(n,d) B(m,d) -> (n,m)"""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0)


def fit_lda(H, y, d, lam=1e-3):
    if os.environ.get("NO_PROJ"):
        # 去掉判别投影：坐标即原特征，维数等于特征维数
        return (np.zeros(H.shape[1], np.float32),
                np.eye(H.shape[1], dtype=np.float32), int(H.shape[1]))
    mu = H.mean(0).astype(np.float64)
    X = H.astype(np.float64) - mu
    cls = np.unique(y)
    K, D = len(cls), H.shape[1]
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in cls:
        m = y == c
        Xc = X[m]
        mc = Xc.mean(0)
        Sw += Xc.T @ Xc - len(Xc) * np.outer(mc, mc)
        Sb += len(Xc) * np.outer(mc, mc)
    Sw /= max(1, len(X) - K)
    Sb /= max(1, K - 1)
    tr = np.trace(Sw) / D
    Sw_r = Sw + lam * tr * np.eye(D)
    L = np.linalg.cholesky(Sw_r)
    Li = np.linalg.inv(L)
    A = Li @ Sb @ Li.T
    A = (A + A.T) / 2
    w, U = np.linalg.eigh(A)
    d_eff = min(d, K - 1)
    idx = np.argsort(w)[::-1][:d_eff]
    W = Li.T @ U[:, idx]
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return mu.astype(np.float32), W.astype(np.float32), d_eff


def knn_graph(Z, k, cls=None, cap=0, tiered=None, tier_cap=0, block=1500):
    """对称化邻居表，返回邻居表与度

    cap>0：邻居按类去重，每类最多 cap 个（exp34 口径）。
    cap=0：纯 kNN。
    tiered=(kn, km, kf, mid_lo, far_lo)：近中远三档。近档取最近 kn 个不限类，
    中档从第 kn~mid_lo 名里按类去重取 km 个，远档从第 far_lo 名起按类去重取 kf 个。
    tier_cap>0：近档也按类去重，每类最多 tier_cap 个（与三档叠加的口径）；
    去重后不足 kn 个就保持不足，不向外补，以反映名额被限制的代价。
    """
    M = len(Z)
    if tiered:
        want = min(M - 1, max(tiered[3] * 3, 600))
    elif cap == 0:
        want = k
    else:
        want = min(M - 1, max(k * 8, 64))
    nb = np.full((M, k), -1, np.int32)
    for i in range(0, M, block):
        j = min(i + block, M)
        Dm = sqdist(Z[i:j], Z)
        Dm[np.arange(j - i), np.arange(i, j)] = np.inf
        part = np.argpartition(Dm, min(want, M - 2), axis=1)[:, :want]
        rows = np.arange(j - i)[:, None]
        order = np.argsort(Dm[rows, part], axis=1)
        part = np.take_along_axis(part, order, axis=1)
        if tiered:
            kn, km, kf, mid_lo, far_lo = tiered
            for r in range(j - i):
                p = part[r]
                if tier_cap > 0:
                    cnt0, sel = {}, []
                    for t in p[:kn]:
                        c0 = int(cls[t])
                        if cnt0.get(c0, 0) < tier_cap:
                            cnt0[c0] = cnt0.get(c0, 0) + 1
                            sel.append(int(t))
                else:
                    sel = [int(t) for t in p[:kn]]
                seen = set(int(cls[t]) for t in sel)
                cnt = 0
                for t in p[kn:mid_lo]:
                    c = int(cls[t])
                    if c in seen:
                        continue
                    seen.add(c)
                    sel.append(int(t))
                    cnt += 1
                    if cnt >= km:
                        break
                cnt = 0
                for t in p[far_lo:]:
                    c = int(cls[t])
                    if c in seen:
                        continue
                    seen.add(c)
                    sel.append(int(t))
                    cnt += 1
                    if cnt >= kf:
                        break
                nb[i + r, :min(len(sel), k)] = sel[:k]
        elif cap == 0:
            nb[i:j] = part[:, :k]
        else:
            for r in range(j - i):
                cnt, sel = {}, []
                for t in part[r]:
                    c = int(cls[t])
                    if cnt.get(c, 0) < cap:
                        cnt[c] = cnt.get(c, 0) + 1
                        sel.append(int(t))
                        if len(sel) >= k:
                            break
                nb[i + r, :len(sel)] = sel
        if (i // block) % 5 == 0:
            log(f"    kNN {j}/{M}")
    # 对称化：边 (i,j) 双向
    adj = [[] for _ in range(M)]
    for i in range(M):
        for j in nb[i]:
            if j >= 0:
                adj[i].append(int(j))
    for i in range(M):
        for j in adj[i]:
            adj[j].append(i)
    deg = np.array([len(a) for a in adj])
    Kmax = max(deg.max(), 1)
    table = np.full((M, Kmax), -1, np.int32)
    for i, a in enumerate(adj):
        lst = sorted(set(a))
        table[i, :len(lst)] = lst
    return table, deg


def greedy_route(Dq_row, table, start, max_hop=30):
    """在距离行向量 Dq_row 上做贪婪下降，返回路径"""
    cur = start
    path = [cur]
    for _ in range(max_hop):
        cand = table[cur]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            break
        dd = Dq_row[cand]
        j = int(np.argmin(dd))
        if dd[j] < Dq_row[cur]:
            cur = int(cand[j])
            path.append(cur)
        else:
            break
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="in1k", choices=["in1k", "c100"])
    ap.add_argument("--R", type=int, default=35, help="每类副本数")
    ap.add_argument("--nq", type=int, default=3000, help="查询条数")
    ap.add_argument("--nstart", type=int, default=3, help="并行起点数")
    ap.add_argument("--pool", type=int, default=0, help="每类用于算原型的样本数，0=自动")
    ap.add_argument("--proj", default="lda", choices=["lda", "lda128", "raw"])
    ap.add_argument("--tiered", default="", help="近中远三档，格式 kn,km,kf,mid_lo,far_lo")
    ap.add_argument("--vnode", default="mean", choices=["mean", "sample"],
                    help="mean=子原型均值（正文 p_n,c 的定义）；sample=真实样本（表 6-9 口径）")
    ap.add_argument("--cap", type=int, default=0, help="近邻按类去重，每类最多 cap 个；0=纯 kNN")
    ap.add_argument("--tier-cap", type=int, default=0,
                    help="与三档叠加：近档也按类去重，每类最多 tier_cap 个；0=近档不限类")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.data == "in1k":
        H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
        y = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int32)
    else:
        H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
        y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)

    K = int(y.max()) + 1
    log(f"  数据 {args.data}: {H.shape}, K={K}")

    # 每类划分：原型池 / 查询池
    idx_by_c = [np.where(y == c)[0] for c in range(K)]
    n_min = min(len(v) for v in idx_by_c)
    n_pool = args.pool or max(20, int(n_min * 0.6))
    proto_idx, query_idx = [], []
    for c in range(K):
        v = idx_by_c[c]
        rng.shuffle(v)
        proto_idx.append(v[:n_pool])
        query_idx.append(v[n_pool:])
    proto_idx = np.concatenate(proto_idx)
    query_idx = np.concatenate(query_idx)
    log(f"  原型池 {len(proto_idx)}，查询池 {len(query_idx)}，每类原型 {n_pool}")

    # 投影
    if args.proj == "raw":
        mu = H[proto_idx].mean(0).astype(np.float32)
        W = np.eye(H.shape[1], dtype=np.float32)
        d = H.shape[1]
        log(f"  不投影，仅中心化，d={d}（K-1={K-1}）")
    else:
        d_req = 128 if args.proj == "lda128" else 384
        mu, W, d = fit_lda(H[proto_idx], y[proto_idx], d_req)
        log(f"  LDA d={d}（请求 {d_req}，K-1={K-1}）")
    Zp = (H[proto_idx] - mu) @ W
    Zq_all = (H[query_idx] - mu) @ W

    # 子原型：每类 R 份
    M = K * args.R
    Zv = np.zeros((M, d), np.float32)
    vcls = np.zeros(M, np.int32)
    vid = 0
    n_sub = max(4, int(n_pool * 0.5))
    for c in range(K):
        rows = H[proto_idx[y[proto_idx] == c]]
        if args.vnode == "sample":
            # 真实样本作 vnode：从该类原型池中随机抽 R 个（有放回）
            pick_rows = rows[rng.choice(len(rows), args.R, replace=True)]
            Zv[vid:vid + args.R] = (pick_rows - mu) @ W
            vcls[vid:vid + args.R] = c
            vid += args.R
        else:
            for _ in range(args.R):
                pick = rng.choice(len(rows), n_sub, replace=True)
                Zv[vid] = (rows[pick].mean(0) - mu) @ W
                vcls[vid] = c
                vid += 1
    log(f"  vnode M={M}，每类 {args.R} 副本，构造方式 {args.vnode}"
        + (f"，每副 {n_sub} 张" if args.vnode == "mean" else ""))

    # 近邻表长度
    k = int(round(max(8, 6 + 4 * np.log10(M / 100))))
    tier_cfg = tuple(int(v) for v in args.tiered.split(",")) if args.tiered else None
    table, deg = knn_graph(Zv, k, cls=vcls, cap=args.cap, tiered=tier_cfg,
                           tier_cap=args.tier_cap)
    k_real = table.shape[1]
    log(f"  近邻表 k={k}，"
        + (f"三档 {tier_cfg}" + (f"，近档每类至多 {args.tier_cap}" if args.tier_cap else "")
           if tier_cfg else f"按类去重 {'关' if args.cap == 0 else args.cap}")
        + f"，实测平均度 {deg.mean():.2f}，最长 {k_real}")

    # 查询抽样
    nq = min(args.nq, len(query_idx))
    qi = rng.choice(len(query_idx), nq, replace=False)
    Q = Zq_all[qi]
    yq = y[query_idx][qi]

    # 距离表（查询 x vnode）
    Dqv = np.sqrt(sqdist(Q, Zv))
    log(f"  距离表 {Dqv.shape}")

    res = {"data": args.data, "K": K, "R": args.R, "M": M, "d": d,
           "k": k, "mean_deg": float(deg.mean()), "nq": nq,
           "proj": args.proj, "n_pool": n_pool, "n_sub": n_sub}

    # 单起点：8 个随机里最近
    n_cand = max(8, args.nstart * 2)
    starts = np.stack([rng.choice(M, n_cand, replace=False) for _ in range(nq)])
    starts = np.sort(starts, 1)
    rows = np.arange(nq)[:, None]
    d0 = Dqv[rows, starts]
    first = starts[rows[:, 0], np.argmin(d0, 1)]

    paths1 = [greedy_route(Dqv[i], table, int(first[i])) for i in range(nq)]
    term1 = np.array([p[-1] for p in paths1])
    hop1 = np.array([len(p) - 1 for p in paths1])
    A1 = float((vcls[term1] == yq).mean())

    # 3 起点：取最近的 3 个互不同类的起点
    res3 = []
    for i in range(nq):
        order = np.argsort(d0[i])
        picked = []
        seen = set()
        for o in order:
            s = int(starts[i, o])
            c = int(vcls[s])
            if c in seen:
                continue
            seen.add(c)
            picked.append(s)
            if len(picked) == args.nstart:
                break
        paths = [greedy_route(Dqv[i], table, s) for s in picked]
        res3.append(paths)
    term3 = [np.array([p[-1] for p in ps]) for ps in res3]
    hop3 = np.array([sum(len(p) - 1 for p in ps) for ps in res3])
    A3 = float(np.mean([any(vcls[t] == yq[i] for t in term3[i]) for i in range(nq)]))
    # 3 起点按类去重后的 top-3 类（按距离排序）
    top3 = []
    for i in range(nq):
        cand = np.unique(term3[i])
        cand = cand[np.argsort(Dqv[i, cand])][:3]
        top3.append(set(vcls[cand].tolist()))
    hit3 = float(np.mean([yq[i] in top3[i] for i in range(nq)]))

    # 1 跳扩散：单起点终点 + 其邻居
    A1d, hit1d, extra1d = [], [], []
    for i in range(nq):
        t = term1[i]
        cand = table[t]
        cand = cand[cand >= 0]
        cand = np.unique(np.concatenate([[t], cand]))
        cand = cand[np.argsort(Dqv[i, cand])][:3]
        s = set(vcls[cand].tolist())
        hit1d.append(yq[i] in s)
        A1d.append(vcls[t] == yq[i])
        extra1d.append(len(s))
    hit1d = float(np.mean(hit1d))
    extra1d = float(np.mean(extra1d))

    # oracle：全局最近 vnode 的类，以及类级最近 3 类（特征口径的上限）
    best = Dqv.argmin(1)
    oracle1 = float((vcls[best] == yq).mean())
    Dqc = np.full((nq, K), np.inf, np.float32)
    for c in range(K):
        rows_c = np.where(vcls == c)[0]
        if len(rows_c):
            Dqc[:, c] = Dqv[:, rows_c].min(1)
    order_c = np.argsort(Dqc, 1)[:, :3]
    oracle3 = float(np.mean([yq[i] in order_c[i] for i in range(nq)]))

    # 副本重合：同类副本间的平均距离 vs 类间最近距离中位数
    intra = []
    for c in range(K):
        rows_c = np.where(vcls == c)[0]
        sub = Zv[rows_c]
        if len(sub) > 1:
            dd = np.sqrt(sqdist(sub, sub))
            iu = np.triu_indices(len(sub), 1)
            intra.append(dd[iu].mean())
    intra = float(np.mean(intra))
    # 类原型之间的最近距离中位数（类间距的保守读法）
    Cm = np.stack([Zv[vcls == c].mean(0) for c in range(K)])
    Dc = np.sqrt(sqdist(Cm, Cm))
    np.fill_diagonal(Dc, np.inf)
    inter = float(np.median(Dc.min(1)))
    # 邻居中同类占比
    same_cls = []
    for i in range(0, min(M, 3000)):
        cand = table[i]
        cand = cand[cand >= 0]
        if len(cand):
            same_cls.append((vcls[cand] == vcls[i]).mean())
    same_cls = float(np.mean(same_cls))

    res.update({
        "vnode": args.vnode, "cap": args.cap, "tier_cap": args.tier_cap,
        "oracle_1nn": oracle1, "oracle_class3": oracle3,
        "A_single": A1, "hop_single": float(hop1.mean()),
        "A_3start": A3, "hop_3start": float(hop3.mean()),
        "hit3_3start_dedup": hit3,
        "hit3_1hop": hit1d, "n_cls_1hop": extra1d,
        "intra_replica_dist": intra, "inter_class_dist": inter,
        "intra_over_inter": intra / max(inter, 1e-9),
        "nbr_same_class_frac": same_cls,
        "cross_edge_frac": 1.0 - same_cls,
        "bytes_coord": 4 * d, "bytes_fwd": 4 * d + ELL,
        "bytes_table": int(k_real * (4 * d + ELL)),
        "bytes_table_4cls": int(4 * k_real * (4 * d + ELL)),
        "bytes_all_coords": int(M * 4 * d),
        "W_bytes": int(4 * 384 * d),
    })
    log("  " + json.dumps(res, ensure_ascii=False, indent=1))

    out = os.path.join(ROOT, "results", f"results_scale_{args.data}_R{args.R}_{args.proj}_{args.vnode}"
                             f"_cap{args.cap}_s{args.nstart}" + (f"_tier{args.tiered.replace(',','-')}" if args.tiered else "") + (f"_tcap{args.tier_cap}" if args.tier_cap else "") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"  写入 {out}")


if __name__ == "__main__":
    main()
