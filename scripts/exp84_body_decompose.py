"""exp84: 四档设定下 body 对判别的真实作用，按 A·S_exp+(1-A)·eps 分解

端到端精度混合了路由与判别，看不出 body 改的是哪一项。本脚本对每档只训
E0（无 body）与 E1（body 本地），把端到端拆成三项：

  A      路由 top-1，送对的概率
  S_exp  送对时本地判对的概率，body 该改善的就是这一项
  eps    送错时蒙对的概率，判别越锐利这一项通常越低

设定（均为坐标取 φ(x) 384 维原样）：
  c10        20 节点 × 2 类，每类 300 张，600 张/节点
  c100       200 节点 × 2 类，每类 125 张，250 张/节点
  c100_20c   5 节点 × 20 类，每类 300 张，6000 张/节点
  in1k_1k    50 节点 × 20 类，每类 25 张，500 张/节点
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp81_body_np import (init_params, train, predict,  # noqa: E402
                           HID, SEED, EP_JOINT)


def log(*a):
    print(*a, flush=True)


def load_c10():
    from exp12_proto_route import build_split
    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float64)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float64)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    if len(T) != len(y_te):
        T = T[te_sel]
    return H, y_tr, tr_node, T, y_te, [sorted(x) for x in nc], C["n_nodes"]


def load_c100():
    import pandas as pd
    from exp1_route import make_shard2
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float64)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float64)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr = d["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte = te["fine_label"].to_numpy().astype(np.int64)
    n = min(len(Htr), len(ytr)); Htr, ytr = Htr[:n], ytr[:n]
    n = min(len(Hte), len(yte)); Hte, yte = Hte[:n], yte[:n]
    nc, ni = make_shard2(ytr, 200, 100, 2, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for i, idx in enumerate(ni):
        for c in nc[i]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), i))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    return Htr[tr_sel], ytr[tr_sel], tr_node, Hte[:3000], yte[:3000], [sorted(x) for x in nc], 200


def load_c100_20c():
    import pandas as pd
    from exp1_route import make_shard2
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float64)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float64)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr = d["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte = te["fine_label"].to_numpy().astype(np.int64)
    n = min(len(Htr), len(ytr)); Htr, ytr = Htr[:n], ytr[:n]
    n = min(len(Hte), len(yte)); Hte, yte = Hte[:n], yte[:n]
    nc, ni = make_shard2(ytr, 5, 100, 20, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for i, idx in enumerate(ni):
        for c in nc[i]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), i))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    return Htr[tr_sel], ytr[tr_sel], tr_node, Hte[:3000], yte[:3000], [sorted(x) for x in nc], 5


def load_in1k_1k():
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float64)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    tr_idx, te_idx = [], []
    for c in range(1000):
        rows = np.where(yt == c)[0]
        tr_idx.append(rows[:25])
        te_idx.append(rows[25:])
    H, y = Ht[np.concatenate(tr_idx)], yt[np.concatenate(tr_idx)]
    T, yT = Ht[np.concatenate(te_idx)], yt[np.concatenate(te_idx)]
    nc = [list(map(int, a)) for a in np.array_split(np.arange(1000), 50)]
    return H, y, np.repeat(np.arange(50), 0), T, yT, nc, 50


def run(name, loader):
    t0 = time.time()
    H, y, _, T, yT, nc, NN = loader()
    if name == "in1k_1k":
        # 千类档每类只有 1 个副本，节点归属由类区间直接给出
        y2node = {}
        for i in range(NN):
            for c in nc[i]:
                y2node[c] = i
        tr_node = np.array([y2node[c] for c in y], dtype=np.int64)
        H2, y2 = H, y
    else:
        H2, y2 = H, y
        tr_node = _
    log(f"[{name}] 训练 {H2.shape} 测试 {T.shape} 节点 {NN} 每节点 {len(nc[0])} 类，"
        f"每节点样本约 {len(H2)//NN}")

    protos, pcls, pnode = [], [], []
    for i in range(NN):
        for c in nc[i]:
            protos.append(H2[y2 == c].mean(0))
            pcls.append(c)
            pnode.append(i)
    P = np.stack(protos)
    pcls, pnode = np.array(pcls), np.array(pnode)

    Dm = ((T ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2 * T @ P.T)
    np.maximum(Dm, 0, out=Dm)
    top1 = np.argmin(Dm, 1)
    top_node, top_cls = pnode[top1], pcls[top1]
    A = float(np.mean(top_cls == yT))
    del Dm
    right = top_cls == yT
    log(f"vnode {len(P)}  A {A:.4f}   用时 {time.time()-t0:.0f}s")

    Hn, yn = {}, {}
    for i in range(NN):
        m = tr_node == i
        Hn[i] = H2[m]
        yn[i] = np.array([nc[i].index(c) for c in y2[m]], dtype=np.int64)

    res = {"A": A, "n_nodes": NN, "per_node_classes": len(nc[0]),
           "per_node_samples": int(len(H2) // NN), "n_test": int(len(T))}

    def evaluate(tag, ps):
        hit = np.zeros(len(T), bool)
        for i in range(NN):
            m = top_node == i
            if m.sum() == 0:
                continue
            pred = np.array(nc[i])[predict(ps[i], T[m])]
            hit[m] = pred == yT[m]
        s_exp = float(hit[right].mean())
        eps = float(hit[~right].mean()) if (~right).sum() else 0.0
        e2e = float(hit.mean())
        res[tag] = {"D1": e2e, "S_exp": s_exp, "eps": eps}
        log(f"  {tag:12s} D1 {e2e:.4f}  S_exp {s_exp:.4f}  eps {eps:.4f}  "
            f"项数 {int(right.sum())}/{int((~right).sum())}   用时 {time.time()-t0:.0f}s")

    for tag, hid in (("E0", None), ("E1", HID)):
        rng = np.random.RandomState(SEED)
        ps = {}
        for i in range(NN):
            ps[i] = init_params(384, hid, len(nc[i]), rng)
            train(ps[i], Hn[i], yn[i], len(nc[i]), EP_JOINT)
        evaluate(tag, ps)
    res["dS_exp_pt"] = round((res["E1"]["S_exp"] - res["E0"]["S_exp"]) * 100, 2)
    res["deps_pt"] = round((res["E1"]["eps"] - res["E0"]["eps"]) * 100, 2)
    res["dD1_pt"] = round((res["E1"]["D1"] - res["E0"]["D1"]) * 100, 2)
    log(f"  → S_exp {res['dS_exp_pt']:+.2f} 点，eps {res['deps_pt']:+.2f} 点，"
        f"端到端 {res['dD1_pt']:+.2f} 点")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    table = {
        "c10": load_c10, "c100": load_c100,
        "c100_20c": load_c100_20c, "in1k_1k": load_in1k_1k,
    }
    names = list(table) if which == "all" else [which]
    out = {}
    for nm in names:
        out[nm] = run(nm, table[nm])
        with open(os.path.join(ROOT, "results", "results_body_decompose.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    log("写入 results_body_decompose.json")
