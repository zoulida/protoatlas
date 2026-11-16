"""exp32: 弱特征下会不会翻盘？—— 原始像素 vs DINOv2，同一套四方对比

exp29 的结构性不等式：R × S > G 才赢。
本案 DINOv2：G=0.963, S=0.997, R=0.942 → 0.939 < 0.963（输）
推测：特征变弱时 G 掉得快、S 掉得慢，交叉点会出现。

本实验把 φ 换成"原始像素 3072 维"（没有任何预训练），重跑：
  G  集中式：一个模型分 10 类
  S  专家  ：每节点只分自己 2 类
  R  路由  ：PCA 坐标 + 原型最近邻
  端到端 = R × S
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


def train_linear(X, y, n_cls, epochs=300, lr=0.05, wd=1e-4):
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
    return m


def pred(m, X, bs=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def run(tag, H, T, y_tr, y_te, nc, tr_node, Nn, cls_of_node):
    log("\n" + "=" * 70)
    log(f"=== {tag} ===  特征维度 {H.shape[1]}")

    # G 集中式
    mG = train_linear(H, y_tr, 10)
    G = float(np.mean(pred(mG, T).argmax(1) == y_te))

    # S 专家（每节点只分自己 2 类，在"属于自己类的样本"上测）
    heads, S_list = {}, []
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]))
    for n in range(Nn):
        idx = np.where(np.isin(y_te, cls_of_node[n]))[0]
        if len(idx) == 0:
            continue
        p = pred(heads[n], T[idx]).argmax(1)
        S_list.append(float(np.mean(np.array(cls_of_node[n])[p] == y_te[idx])))
    S = float(np.mean(S_list))

    # R 路由（PCA 坐标 + 原型最近邻）
    mu = H.mean(0, keepdims=True)
    Xc = H - mu
    Cc = (Xc.T @ Xc) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    Zp = (P - mu[0]) @ W
    Zq = (T - mu[0]) @ W
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2 * Zq @ Zp.T, 0))
    R = float(np.mean(pcls[D.argmin(1)] == y_te))

    # 端到端：走路由，用宿主节点作答
    o = D.argmin(1)
    pr = np.zeros(len(T), np.int32)
    for n in np.unique(pnode[o]):
        idx = np.where(pnode[o] == n)[0]
        pr[idx] = np.array(cls_of_node[n])[pred(heads[n], T[idx]).argmax(1)]
    E = float(np.mean(pr == y_te))

    log(f"  G 集中式（分 10 类）        {G:.4f}")
    log(f"  S 专家（分自己 2 类）        {S:.4f}   ← 比 G {'高' if S>G else '低'} {abs(S-G)*100:.1f} 点")
    log(f"  R 路由命中                  {R:.4f}")
    log(f"  R×S（理论上端到端）          {R*S:.4f}")
    log(f"  实际端到端（含 B 象限补偿）    {E:.4f}")
    log(f"  → {'赢' if E > G else '输'} {abs(E-G)*100:.1f} 点")
    return dict(G=G, S=S, R=R, RS=R * S, E=E, dim=int(H.shape[1]))


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    # DINOv2
    Hd = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    Td = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)

    # 原始像素
    d = np.load(os.path.join(CACHE, "cifar10.npz"))
    Xr = d["xtr"][tr_sel].reshape(-1, 3072).astype(np.float32) / 255.0
    Xe = d["xte"][te_sel].reshape(-1, 3072).astype(np.float32) / 255.0
    m_, s_ = Xr.mean(0, keepdims=True), Xr.std(0, keepdims=True) + 1e-6
    Xr = (Xr - m_) / s_
    Xe = (Xe - m_) / s_

    res = {}
    res["dinov2"] = run("DINOv2 ViT-S/14（强特征）", Hd, Td, y_tr, y_te,
                        nc, tr_node, Nn, cls_of_node)
    res["rawpixel"] = run("原始像素 3072 维（无预训练）", Xr, Xe, y_tr, y_te,
                          nc, tr_node, Nn, cls_of_node)

    with open(os.path.join(ROOT, "results", "results_weak_feat.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 70)
    log("  判读：若原始像素下 E > G，说明'特征越弱，路由越有价值'成立，")
    log("        论文主战场应放在弱 φ / 异构数据（金融 tabular、FEMNIST）")


if __name__ == "__main__":
    main()
