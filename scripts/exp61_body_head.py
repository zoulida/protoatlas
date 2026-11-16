"""exp61: 放开 body —— body 与 head 分开训练对端到端精度的影响

方法设定：φ 是协议资产（全网同一版本、不参与训练），body 接在 φ 之后、
各节点自己训练，head 本地训练。坐标 z=(φ(x)-μ)W 只由 φ 决定，故 body 的
差异不进入路由，路由侧与 exp29b 完全一致，本脚本只替换每个节点的判别器。

四档（判别器不同，候选列表与路由完全相同）：
  E0  body 取恒等（正文现读数，等价于没有 body），head 直接接 384 维
  E1  body = 384→256→ReLU，本地联合训练，不同步
  E2  body 同 E1，但在坐标最近的 k 个邻居间同步 3 轮（模拟邻近共同训练）
  E3  body 同 E1，但在全网同步 3 轮（上界参照，对应 FedAvg 式的 body 共享）

E2/E3 每轮：本地训练 body+head → 平均 body（head 不平均）；最后一轮之后
冻结 body、用最终 body 重训 head，保证 head 适配同步后的 body。

主指标 D1 = top-1 候选的节点作答后的端到端精度；S = 各节点本地判别的平均精度。
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
torch.manual_seed(0)
np.random.seed(0)


def log(*a):
    print(*a, flush=True)


def fit_lda(H, y, d, lam=1e-3):
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


def build(hid, kin, d=384):
    if hid is None:
        body = nn.Identity()
        hd = d
    else:
        body = nn.Sequential(nn.Linear(d, hid), nn.ReLU())
        hd = hid
    return body, nn.Linear(hd, kin)


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


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"数据：训练 {H.shape} 测试 {T.shape} 节点 {Nn}")

    # ---------- 路由侧：与 exp29b 的 single 配置一致，四档共用
    mu, W, d_eff = fit_lda(H, y_tr, 9)
    log(f"判别投影 d={d_eff}")
    Zq = (T - mu) @ W

    protos, pcls, pnode, node_z = [], [], [], {}
    for n in range(Nn):
        m = tr_node == n
        zs = []
        for c in nc[n]:
            X = H[m & (y_tr == c)]
            if len(X) == 0:
                continue
            p = X.mean(0)
            protos.append(p)
            pcls.append(c)
            pnode.append(n)
            zs.append((p - mu) @ W)
        node_z[n] = np.mean(zs, 0)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zp = (P - mu) @ W
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
    log(f"候选召回@3 {in_cand:.4f}（四档相同）")

    NZ = np.stack([node_z[n] for n in range(Nn)])
    dnn = np.sqrt(((NZ[:, None, :] - NZ[None, :, :]) ** 2).sum(-1))
    nbr_all = {n: list(np.argsort(dnn[n])[:8]) for n in range(Nn)}

    Hn, yn = {}, {}
    for n in range(Nn):
        m = tr_node == n
        Hn[n] = H[m]
        yn[n] = np.array([cls_of_node[n].index(c) for c in y_tr[m]], dtype=np.int64)

    res = {"d": int(d_eff), "cand_recall3": in_cand,
           "msg_body_kb": 384 * 256 * 4 / 1024 + 256 * 2 * 4 / 1024}

    def evaluate(tag, bodies, heads):
        pred_all = {}
        for n in range(Nn):
            pred_all[n] = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T).argmax(1)]
        d1 = np.array([pred_all[cl[i][0][1]][i] for i in range(len(T))])
        d3 = np.array([any(pred_all[n][i] == y_te[i] for c, n, _ in cl[i])
                       for i in range(len(T))])
        s = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))
        res[tag] = dict(D1=float(np.mean(d1 == y_te)), D3_local=float(np.mean(d3)),
                        S=s)
        log(f"{tag:28s} D1 {res[tag]['D1']:.4f}  候选内本地答对 {res[tag]['D3_local']:.4f}  "
            f"各节点全局精度均值 {s:.4f}")

    # ---------- E0：body 取恒等
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(None, len(cls_of_node[n]))
        train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
    evaluate("E0 无 body（385 维直连）".replace("385", "384"), bodies, heads)

    # ---------- E1：body 本地训练，不同步
    bodies, heads = {}, {}
    for n in range(Nn):
        bodies[n], heads[n] = build(256, len(cls_of_node[n]))
        train_join(Hn[n], yn[n], bodies[n], heads[n], 400)
    evaluate("E1 body 本地训练", bodies, heads)

    # ---------- E2/E3：body 同步
    def sync_run(tag, nbrs, rounds=3, ep=200):
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
            # 平均之后 head 面对的是新的 body，重训 head
            for n in range(Nn):
                heads[n] = build(256, len(cls_of_node[n]))[1]
                train_head_only(Hn[n], yn[n], bodies[n], heads[n], 300)
            log(f"   {tag} 第 {r + 1} 轮完成")
        evaluate(tag, bodies, heads)

    sync_run("E2 body 近邻同步(8 邻居×3 轮)", nbr_all)
    sync_run("E3 body 全网同步(20 节点×3 轮)", {n: list(range(Nn)) for n in range(Nn)})

    with open(os.path.join(ROOT, "results", "results_body_head.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_body_head.json")


if __name__ == "__main__":
    main()
