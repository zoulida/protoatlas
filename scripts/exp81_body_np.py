"""exp81: 删去判别投影后，body 与 head 分开训练还有没有增益（纯 numpy 版）

坐标 z = φ(x)，384 维原样，不作中心化也不作投影。
路由、候选、邻居四档完全相同，只换每个节点的判别器：

  E0  body 取恒等，head 直接接 384 维（正文现读数）
  E1  body = 384→256→ReLU，本地联合训练，不同步
  E2  body 同 E1，在坐标最近的 8 个节点间同步 3 轮（簇内共同训练）
  E3  body 同 E1，在全网同步 3 轮（上界参照，FedAvg 式共享）

训练口径与 exp61/exp62 相同：Adam，lr 5e-3，weight decay 1e-4，全批量，
联合训练 400 步；同步档每轮本地 200 步，平均 body 后用 300 步重训 head。

读两个量：D1 为 top-1 候选由宿主节点作答后的端到端精度，S 为各节点本地精度的均值。

用法：python exp81_body_np.py c10|c100
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

SEED = 0
HID = 256
LR = 5e-3
WD = 1e-4
EP_JOINT = 400
EP_SYNC = 200
EP_HEAD = 300
ROUNDS = 3


def log(*a):
    print(*a, flush=True)


def init_params(d_in, hid, d_out, rng):
    if hid is None:
        return dict(W1=np.zeros((d_in, 0), np.float64), b1=np.zeros(0),
                    W2=rng.normal(0, np.sqrt(1.0 / d_in), (d_in, d_out)), b2=np.zeros(d_out))
    return dict(
        W1=rng.normal(0, np.sqrt(2.0 / d_in), (d_in, hid)),
        b1=np.zeros(hid),
        W2=rng.normal(0, np.sqrt(1.0 / hid), (hid, d_out)),
        b2=np.zeros(d_out))


def forward(p, X):
    if p["W1"].shape[1] == 0:
        return X, X @ p["W2"] + p["b2"]
    F = X @ p["W1"] + p["b1"]
    np.maximum(F, 0, out=F)
    return F, F @ p["W2"] + p["b2"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_grad(p, X, y, K, train_body=True):
    F, logit = forward(p, X)
    prob = softmax(logit)
    n = len(y)
    ce = -np.log(np.maximum(prob[np.arange(n), y], 1e-12)).mean()
    g = prob.copy()
    g[np.arange(n), y] -= 1
    g /= n
    grads = dict(W2=F.T @ g + WD * p["W2"], b2=g.sum(0))
    loss = ce
    if train_body and p["W1"].shape[1] > 0:
        dF = (g @ p["W2"].T) * (F > 0)
        grads["W1"] = X.T @ dF + WD * p["W1"]
        grads["b1"] = dF.sum(0)
        loss += 0.5 * WD * (p["W1"] ** 2).sum()
    loss += 0.5 * WD * (p["W2"] ** 2).sum()
    return loss, grads


class Adam:
    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(p, X, y, K, steps, train_body=True):
    keys = ["W2", "b2"] + (["W1", "b1"] if train_body and p["W1"].shape[1] > 0 else [])
    opt = Adam({k: p[k] for k in keys})
    for _ in range(steps):
        _, g = loss_grad(p, X, y, K, train_body)
        opt.step(p, {k: g[k] for k in keys})


def train_head_only(p, X, y, K, steps):
    keys = ["W2", "b2"]
    opt = Adam({k: p[k] for k in keys})
    for _ in range(steps):
        # 只对 head 求梯度：body 固定
        F2, logit = forward(p, X)
        prob = softmax(logit)
        n = len(y)
        g = prob.copy()
        g[np.arange(n), y] -= 1
        g /= n
        grads = dict(W2=F2.T @ g + WD * p["W2"], b2=g.sum(0))
        opt.step(p, grads)


def predict(p, X):
    _, logit = forward(p, X)
    return logit.argmax(1)


def get_data(name):
    if name == "c10":
        from exp12_proto_route import build_split
        nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
        H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float64)
        T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float64)
        y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
        if len(T) != len(y_te):
            T = T[te_sel]
        return H, y_tr, tr_node, T, y_te, [sorted(x) for x in nc], C["n_nodes"]
    import pandas as pd
    from exp1_route import make_shard2
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float64)
    Hte = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float64)
    d = np.load(os.path.join(CACHE, "c100_train.npz"))
    ytr = d["y"].astype(np.int64)
    te = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    yte = te["fine_label"].to_numpy().astype(np.int64)
    n_te = min(len(Hte), len(yte))
    Hte, yte = Hte[:n_te], yte[:n_te]
    n_tr = min(len(Htr), len(ytr))
    Htr, ytr = Htr[:n_tr], ytr[:n_tr]
    nc, ni = make_shard2(ytr, 200, 100, 2, seed=0)
    rng = np.random.RandomState(1)
    tr_sel, tr_node = [], []
    for n, idx in enumerate(ni):
        for c in nc[n]:
            sel = idx[ytr[idx] == c]
            take = rng.choice(sel, size=min(300, len(sel)), replace=False)
            tr_sel.append(take)
            tr_node.append(np.full(len(take), n))
    tr_sel = np.concatenate(tr_sel)
    tr_node = np.concatenate(tr_node)
    T, yT = Hte[:3000], yte[:3000]
    return Htr[tr_sel], ytr[tr_sel], tr_node, T, yT, [sorted(x) for x in nc], 200


def run(name):
    H, y_tr, tr_node, T, y_te, nc, Nn = get_data(name)
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}
    log(f"[{name}] 训练 {H.shape} 测试 {T.shape} 节点 {Nn} 每节点 {len(nc[0])} 类")

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
    P = np.stack(protos)
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
                out.append((c, int(pnode[j])))
            if len(out) == k:
                break
        return out

    cl = [cands(i) for i in range(len(T))]
    in_cand = float(np.mean([y_te[i] in [c for c, _ in cl[i]] for i in range(len(T))]))
    top1 = float(np.mean([cl[i][0][0] == y_te[i] for i in range(len(T))]))
    log(f"vnode {len(P)}  候选召回@3 {in_cand:.4f}  路由 top-1 {top1:.4f}   用时 {time.time()-t0:.0f}s")

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
           "body_kb": (384 * HID * 4 + HID * 2 * 4) / 1024}

    def evaluate(tag, params):
        pred_all = {n: np.array(cls_of_node[n])[predict(params[n], T)] for n in range(Nn)}
        d1 = np.array([pred_all[cl[i][0][1]][i] for i in range(len(T))])
        d3 = np.array([any(pred_all[n][i] == y_te[i] for c, n in cl[i])
                       for i in range(len(T))])
        s = float(np.mean([np.mean(pred_all[n] == y_te) for n in range(Nn)]))
        res[tag] = dict(D1=float(np.mean(d1 == y_te)), D3_local=float(np.mean(d3)), S=s)
        log(f"{tag:26s} D1 {res[tag]['D1']:.4f}  候选内命中 {res[tag]['D3_local']:.4f}  "
            f"本地均值 {s:.4f}   用时 {time.time()-t0:.0f}s")

    # E0
    rng = np.random.RandomState(SEED)
    ps = {}
    for n in range(Nn):
        ps[n] = init_params(384, None, len(cls_of_node[n]), rng)
        train(ps[n], Hn[n], yn[n], len(cls_of_node[n]), EP_JOINT)
    evaluate("E0 无 body", ps)

    # E1
    rng = np.random.RandomState(SEED)
    ps = {}
    for n in range(Nn):
        ps[n] = init_params(384, HID, len(cls_of_node[n]), rng)
        train(ps[n], Hn[n], yn[n], len(cls_of_node[n]), EP_JOINT)
    evaluate("E1 body 本地", ps)

    def sync_run(tag, nbrs):
        rng = np.random.RandomState(SEED)
        base = init_params(384, HID, 2, rng)
        ps = {}
        for n in range(Nn):
            ps[n] = init_params(384, HID, len(cls_of_node[n]), np.random.RandomState(SEED))
            ps[n]["W1"] = base["W1"].copy()
            ps[n]["b1"] = base["b1"].copy()
        for r in range(ROUNDS):
            for n in range(Nn):
                train(ps[n], Hn[n], yn[n], len(cls_of_node[n]), EP_SYNC)
            for n in range(Nn):
                W1 = np.mean([ps[j]["W1"] for j in nbrs[n]], 0)
                b1 = np.mean([ps[j]["b1"] for j in nbrs[n]], 0)
                ps[n]["W1"], ps[n]["b1"] = W1, b1
            for n in range(Nn):
                K = len(cls_of_node[n])
                ps[n]["W2"] = np.random.RandomState(SEED + n).normal(
                    0, np.sqrt(1.0 / HID), (HID, K))
                ps[n]["b2"] = np.zeros(K)
                train_head_only(ps[n], Hn[n], yn[n], K, EP_HEAD)
            log(f"   {tag} 第 {r+1} 轮")
        evaluate(tag, ps)

    sync_run("E2 body 近邻同步(8×3)", nbr_all)
    if name == "c10":
        sync_run("E3 body 全网同步(20×3)", {n: list(range(Nn)) for n in range(Nn)})

    out = os.path.join(ROOT, "results", f"results_body_noW_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(out)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "c10")
