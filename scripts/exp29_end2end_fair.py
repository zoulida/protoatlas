"""exp29: 端到端公平对比 —— "路由只有 94%，还不如单一节点"这个质疑是否成立？

四个口径，同一批数据（CIFAR-10，shard-2，12000 训练 / 5000 测试）：
  A 集中式     : 一个节点拿全部数据，训 10 类（这是"单一节点"的真实含义）
  B 纯本地     : 每节点只训自己 2 类，对全部测试集作答（不看路由）
  C FedAvg     : 每节点 10 类输出层、本地只有 2 类数据，参数平均 20 轮
  D 我们的     : 路由 top-3 → 节点 head → 裁决

关键区分：
  hit@1（0.942）是"最近的那一个对"，不是系统上界。
  系统上界 = 候选召回@3（0.994）× 节点在自家类上的准确率（0.997）。
  但从"候选"到"单一答案"要经过裁决，裁决会损失 —— 这个损失必须量出来。
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
    """全批量线性头（确定性，便于复现）"""
    m = nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward()
        opt.step()
    return m


def predict(m, X, bs=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def main():
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel].astype(np.int64), yte[te_sel].astype(np.int64)
    Nn = C["n_nodes"]
    cls_of_node = {n: sorted(nc[n]) for n in range(Nn)}

    res = {}
    log(f"数据：训练 {H.shape} 测试 {T.shape} 节点 {Nn}")

    # ---------------- A 集中式 ----------------
    mA = train_linear(H, y_tr, 10)
    accA = float(np.mean(predict(mA, T).argmax(1) == y_te))
    log(f"\nA 集中式（全量 12000 训 10 类）          准确率 {accA:.4f}")
    res["A_central"] = accA

    # ---------------- B 纯本地（不路由） ----------------
    # 每节点只在自己 2 类上训，输出全局类
    heads = {}
    for n in range(Nn):
        m = tr_node == n
        heads[n] = train_linear(H[m], np.array([cls_of_node[n].index(c) for c in y_tr[m]]),
                                len(cls_of_node[n]))

    def local_pred(n, X):
        lg = predict(heads[n], X)
        return np.array(cls_of_node[n])[lg.argmax(1)], lg.max(1)

    # 随机找一个节点代表"纯本地"（无路由时，你只能问自己）
    accs = []
    for n in range(Nn):
        p, _ = local_pred(n, T)
        accs.append(float(np.mean(p == y_te)))
    accB = float(np.mean(accs))
    log(f"B 纯本地（每节点只答自己 2 类）          准确率 {accB:.4f}"
        f"（范围 {min(accs):.3f}~{max(accs):.3f}）① 结构上只能答 2/10 类")
    res["B_local"] = accB

    # ---------------- C FedAvg ----------------
    torch.manual_seed(0)
    glob = nn.Linear(384, 10)
    gsd = glob.state_dict()
    R, E = 20, 3
    for r in range(R):
        states = []
        for n in range(Nn):
            m = tr_node == n
            Xn = torch.from_numpy(H[m])
            yn = torch.from_numpy(y_tr[m])
            loc = nn.Linear(384, 10)
            loc.load_state_dict(gsd)
            opt = torch.optim.SGD(loc.parameters(), lr=0.05, momentum=0.9)
            lossf = nn.CrossEntropyLoss()
            for _ in range(E):
                opt.zero_grad()
                lossf(loc(Xn), yn).backward()
                opt.step()
            states.append({k: v.clone() for k, v in loc.state_dict().items()})
        gsd = {k: torch.stack([s[k] for s in states]).mean(0) for k in gsd}
        glob.load_state_dict(gsd)
    accC = float(np.mean(predict(glob, T).argmax(1) == y_te))
    log(f"C FedAvg（20 轮 × 3 epoch，10 类全局头）  准确率 {accC:.4f}")
    res["C_fedavg"] = accC

    # ---------------- D 我们的 ----------------
    # 原型 + 坐标（d=16，10 类够用）
    protos, pcls, pnode = [], [], []
    for n in range(Nn):
        m = tr_node == n
        for k in nc[n]:
            protos.append(H[m & (y_tr == k)].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    pcls, pnode = np.array(pcls), np.array(pnode)
    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)
    Zp = (P - mu[0]) @ W
    Zq = (T - mu[0]) @ W
    D = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                           - 2 * Zq @ Zp.T, 0))
    order = np.argsort(D, 1)

    def cands(i, k=3):
        seen, out = set(), []
        for j in order[i]:
            c = int(pcls[j])
            if c not in seen:
                seen.add(c)
                out.append((c, int(pnode[j]), float(D[i, j])))
            if len(out) == k:
                break
        return out

    # 各节点对全部测试集的预测
    pred_all = {}
    for n in range(Nn):
        p, _ = local_pred(n, T)
        pred_all[n] = p

    cand_lists = [cands(i) for i in range(len(T))]
    in_cand = np.array([y_te[i] in [c for c, _, _ in cand_lists[i]] for i in range(len(T))])
    log(f"\nD 候选召回@3（上界）                      {in_cand.mean():.4f}")

    # D1 硬 top-1
    d1 = np.array([pred_all[cand_lists[i][0][1]][i] for i in range(len(T))])
    accD1 = float(np.mean(d1 == y_te))
    # D2 距离加权投票（无参数裁决）
    d2 = []
    for i in range(len(T)):
        cs = cand_lists[i]
        ds = np.array([c[2] for c in cs])
        wts = np.exp(-ds / (ds.mean() + 1e-8))
        sc = np.zeros(10)
        for (c, n, dd), wt in zip(cs, wts):
            sc[pred_all[n][i]] += wt
        d2.append(int(np.argmax(sc)))
    accD2 = float(np.mean(np.array(d2) == y_te))
    # D3 oracle 裁决（知道哪个候选对）→ 裁决上界
    accD3 = float(np.mean([
        any(pred_all[n][i] == y_te[i] for c, n, _ in cand_lists[i])
        for i in range(len(T))]))
    # D4 只取 top-1 候选节点的预测，但若距离异常则拒识（95% 分位）
    log(f"D1 硬 top-1（不给候选）                  准确率 {accD1:.4f}")
    log(f"D2 距离加权投票（3 候选，无参数）         准确率 {accD2:.4f}")
    log(f"D3 oracle 裁决（候选中有一个答对即算对）    准确率 {accD3:.4f}  ← 裁决上界")
    res.update(D1=accD1, D2=accD2, D3_oracle=accD3, cand_recall3=float(in_cand.mean()))

    log("\n" + "=" * 62)
    log("  结论对照")
    log("=" * 62)
    for k, v in [("A 集中式（单一节点全量）", accA), ("B 纯本地（不路由）", accB),
                 ("C FedAvg", accC), ("D1 我们的·硬 top-1", accD1),
                 ("D2 我们的·候选+投票", accD2), ("D3 我们的·裁决上界", accD3)]:
        log(f"  {k:<28}{v:.4f}")

    with open(os.path.join(ROOT, "results", "results_end2end_fair.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
