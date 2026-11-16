"""exp65: CIFAR-10（本文设定，每节点 2 类）下，body 增益随每节点样本量的变化

exp64 在 CIFAR-100（每节点 20 类）上做样本量扫描，增益在 0.27–0.77 点之间
无趋势，说明样本量不是决定因素。本脚本在本文的正式设定（CIFAR-10、每节点
2 类、每类 4 副本）上再扫一遍，只能向下扫——特征缓存本身是每类 300 张。

  shots 每类 10 / 25 / 50 / 100 / 200 / 300 → 每节点 20 / 50 / 100 / 200 /
  400 / 600 张

路由与候选固定不变（坐标只由 φ 决定），只改判别器实际用到的样本数。
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

from exp61_body_head import fit_lda, build, train_join, infer  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

SHOTS = (10, 25, 50, 100, 200, 300)


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
    log(f"数据：训练 {H.shape} 测试 {T.shape} 节点 {Nn}")

    mu, W, d_eff = fit_lda(H, y_tr, 9)
    log(f"判别投影 d={d_eff}")
    Zq = (T - mu) @ W

    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        m = tr_node == n
        for c in nc[n]:
            X = H[m & (y_tr == c)]
            if len(X) == 0:
                continue
            protos.append(X.mean(0))
            pcls.append(c)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zp = (P - mu) @ W
    Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                            - 2 * Zq @ Zp.T, 0))
    order = np.argsort(Dm, 1)
    cl = []
    for i in range(len(T)):
        seen = set()
        for j in order[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.add(c)
            cl.append((c, int(pnode[j])))
            break
    top_node = np.array([n for _, n in cl])
    top_cls = np.array([c for c, _ in cl])
    route = float(np.mean(top_cls == y_te))
    log(f"top-1 路由精度 {route:.4f}（各档共用）")

    rows_of = {n: {c: np.where((tr_node == n) & (y_tr == c))[0] for c in nc[n]}
               for n in range(Nn)}

    res = {}
    for shots in SHOTS:
        rng = np.random.RandomState(11)
        Hn, yn = {}, {}
        for n in range(Nn):
            idxs = []
            for c in nc[n]:
                sel = rows_of[n][c]
                take = rng.choice(sel, size=min(shots, len(sel)), replace=False)
                idxs.append(take)
            idx = np.concatenate(idxs)
            Hn[n] = H[idx]
            yn[n] = np.array([cls_of_node[n].index(c) for c in y_tr[idx]], dtype=np.int64)

        out = {"shots_per_class": shots,
               "samples_per_node": int(np.mean([len(v) for v in yn.values()])),
               "route_top1": route}
        log(f"\n=== 每类 {shots} 张，每节点 {out['samples_per_node']} 张 ===")

        def evaluate(bodies, heads):
            pred = {}
            for n in range(Nn):
                pred[n] = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
            d1 = np.array([pred[top_node[i]][i] for i in range(len(T))])
            return float(np.mean(d1 == y_te))

        for tag, hid in (("E0", None), ("E1", 256)):
            t0 = time.time()
            bodies, heads = {}, {}
            for n in range(Nn):
                bodies[n], heads[n] = build(hid, len(cls_of_node[n]))
                train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
            out[tag] = evaluate(bodies, heads)
            log(f"  {tag} D1 {out[tag]:.4f}   用时 {time.time() - t0:.0f}s")
        out["gain_pt"] = round((out["E1"] - out["E0"]) * 100, 2)
        log(f"  增益 {out['gain_pt']:+.2f} 点")
        res[str(shots)] = out

    with open(os.path.join(ROOT, "results", "results_shots_body_c10.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n写入 results_shots_body_c10.json")


if __name__ == "__main__":
    main()
