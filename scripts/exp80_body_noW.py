"""exp80: 删去判别投影后，body 与 head 分开训练还有没有增益

坐标 z = φ(x)，384 维原样，不作中心化也不作投影（与正文现行做法一致）。
路由、候选、邻居关系四档完全相同，只换每个节点的判别器：

  E0  body 取恒等，head 直接接 384 维（正文现读数）
  E1  body = 384→256→ReLU，本地联合训练，不同步
  E2  body 同 E1，在坐标最近的 8 个节点间同步 3 轮（簇内共同训练）
  E3  body 同 E1，在全网同步 3 轮（上界参照，FedAvg 式共享）

E2/E3 每轮：本地训练 body+head → 平均 body（head 不平均）→ 用新 body 重训 head。

读两个量：D1 为 top-1 候选由宿主节点作答后的端到端精度，S 为各节点本地精度的均值。

用法：python exp80_body_noW.py c10|c100
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(8)


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


def train_head_only(X, y, body, head, epochs, lr=5e-3, wd=1e-4):
    for p in body.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    with torch.no_grad():
        F = body(torch.from_numpy(X))
    yt = torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(F), yt).backward()
        opt.step()
    for p in body.parameters():
        p.requires_grad_(True)


def infer(body, head, X):
    with torch.no_grad():
        return head(body(torch.from_numpy(X))).numpy()


def get_data(name):
    if name == "c10":
        from exp12_proto_route import build_split
        nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
        H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
        T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
        y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
        if len(T) != len(y_te):
            T = T[te_sel]
        return H, y_tr, tr_node, T, y_te, [sorted(x) for x in nc], C["n_nodes"]
    from exp53_c100_end2end_lda import load_c100, build_split as bs100, N_TEST
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    nc, tr_sel, tr_node = bs100(y)
    return H[tr_sel], y[tr_sel], tr_node, T, yT, nc, len(nc)


def run(name):
    H, y_tr, tr_node, T, y_te, nc, Nn = get_data(name)
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"[{name}] 训练 {H.shape} 测试 {T.shape} 节点 {Nn} 每节点类数 {len(nc[0])}")

    t0 = time.time()
    protos, pcls, pnode, node_z = [], [], [], {}
    for n in range(Nn):
        m = tr_node == n
        zs = []
        for c in cls_of_node[n]:
            X = H[m & (y_tr == c)]
            if len(X) == 0:
                continue
            p = X.mean(0)
            protos.append(p)
            pcls.append(c)
            pnode.append(n)
            zs.append(p)
        node_z[n] = np.mean(zs, 0)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zq, Zp = T, P
    Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                            - 2 * Zq @ Zp.T, 0))
    order = np.argsort(Dm, 1)

    def cands(i, k=3):
        seen, out = set(), []
        for j in order[i]:
            c = int(pcls[j])
            if c not in seen:
                seen.add(c)
                out.append((c, int(pnode[j]), float(Dm[i, j])))
            if len(out) == k:
                break
        return out

    cl = [cands(i) for i in range(len(T))]
    in_cand = float(np.mean([y_te[i] in [c for c, _, _ in cl[i]] for i in range(len(T))]))
    top1 = float(np.mean([cl[i][0][0] == y_te[i] for i in range(len(T))]))
    log(f"vnode {len(P)}  候选召回@3 {in_cand:.4f}  路由 top-1 {top1:.4f}")

    NZ = np.stack([node_z[n] for n in range(Nn)])
    dnn = np.sqrt(((NZ[:, None, :] - NZ[None, :, :]) ** 2).sum(-1))
    nbr_all = {n: list(np.argsort(dnn[n])[:8]) for n in range(Nn)}

    Hn, yn = {}, {}
    for n in range(Nn):
        m = tr_node == n
        Hn[n] = H[m]
        yn[n] = np.array([cls_of_node[n].index(c) for c in y_tr[m]], dtype=np.int64)

    res = {"dataset": name, "n_nodes": Nn, "vnode": int(len(P)),
           "cand_recall3": in_cand, "route_top1": top1,
           "body_kb": (384 * 256 * 4 + 256 * 2 * 4) / 1024}

    def evaluate(tag, bodies, heads):
        pred_all = {}
        for n in range(Nn):
            pred_all[n] = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
        d1 = np.array([pred_all[cl[i][0][1]][i] for i in range(len(T))])
        d3 = np.array([any(pred_all[n][i] == y_te[i] for c, n, _ in cl[i])
                       for i in range(len(T))])
        s = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))
        res[tag] = dict(D1=float(np.mean(d1 == y_te)), D3_local=float(np.mean(d3)), S=s)
        log(f"{tag:30s} D1 {res[tag]['D1']:.4f}  候选内命中 {res[tag]['D3_local']:.4f}  "
            f"本地均值 {s:.4f}   用时 {time.time() - t0:.0f}s")

    # E0
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(None, len(cls_of_node[n]))
        train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
    evaluate("E0 无 body（384 维直连）", bodies, heads)

    # E1
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(256, len(cls_of_node[n]))
        train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
    evaluate("E1 body 本地训练", bodies, heads)

    def sync_run(tag, nbrs, rounds=3, ep=200, ep_head=300):
        torch.manual_seed(0)
        bodies, heads = {}, {}
        init = None
        for n in range(Nn):
            bodies[n], heads[n] = build(256, len(cls_of_node[n]))
            if init is None:
                init = {k: v.clone() for k, v in bodies[n].state_dict().items()}
            bodies[n].load_state_dict(init)
        for r in range(rounds):
            for n in range(Nn):
                train_join(Hn[n], yn[n], bodies[n], heads[n], ep)
            for n in range(Nn):
                sd = bodies[n].state_dict()
                avg = {k: torch.stack([bodies[j].state_dict()[k] for j in nbrs[n]]).mean(0)
                       for k in sd}
                bodies[n].load_state_dict(avg)
            for n in range(Nn):
                heads[n] = build(256, len(cls_of_node[n]))[1]
                train_head_only(Hn[n], yn[n], bodies[n], heads[n], ep_head)
            log(f"   {tag} 第 {r + 1} 轮")
        evaluate(tag, bodies, heads)

    sync_run("E2 body 近邻同步(8×3)", nbr_all)
    if name == "c10":
        sync_run("E3 body 全网同步(20×3)", {n: list(range(Nn)) for n in range(Nn)})

    out = os.path.join(ROOT, "results", f"results_body_noW_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(out)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "c10")
