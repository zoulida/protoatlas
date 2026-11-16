"""exp8: 每类一个模型 vs 共享模型；共同训练的价值有多大
Part 1  架构对比：softmax 多类  vs  per-class 一对多(OvR)
Part 2  协作价值：全局模型在"本节点 2 类"上的准确率 vs 局部模型单干
        扫描每类样本数 n，看协作收益随样本量如何衰减
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
torch.set_num_threads(8)


def log(*a):
    print(*a, flush=True)


def load_aligned():
    sys.path.insert(0, ROOT)
    import exp1_route as E
    xtr, ytr, xte, yte = E.load_cifar()
    C = E.CFG
    nc, ni = E.make_shard2(ytr, C["n_nodes"], C["n_classes"],
                           C["shard_per_node"], C["seed"])
    rng = np.random.RandomState(C["seed"] + 1)
    sel, nod = [], []
    for n, idx in enumerate(ni):
        for k in nc[n]:
            sub = idx[ytr[idx] == k]
            take = rng.choice(sub, size=min(C["per_class_train"], len(sub)), replace=False)
            sel.append(take)
            nod.append(np.full(len(take), n))
    tr_sel = np.concatenate(sel)
    tr_node = np.concatenate(nod)
    te_sel = rng.choice(len(xte), size=min(C["n_test"], len(xte)), replace=False)
    return dict(ytr=ytr, y=ytr[tr_sel], tr_sel=tr_sel, tr_node=tr_node,
                nc=nc, yte=yte[te_sel], te_sel=te_sel)


def train_softmax(X, y, ncls, ep=300, lr=0.05, wd=1e-4):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    m = nn.Linear(X.shape[1], ncls)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward()
        opt.step()
    m.eval()
    return m


def pred_logits(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()


def main():
    t0 = time.time()
    P = load_aligned()
    Htr = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    Hte = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y, yte = P["y"], P["yte"]

    # 特征归一化（线性模型必须做，否则不同维度量纲差异巨大）
    mu = Htr.mean(0, keepdims=True)
    sd = Htr.std(0, keepdims=True) + 1e-6
    Ztr = (Htr - mu) / sd
    Zte = (Hte - mu) / sd
    log(f"数据 train{Ztr.shape} test{Zte.shape}  dim={Ztr.shape[1]}")

    out = {}

    # ---------------- Part 1: 架构对比 ----------------
    log("\n===== Part 1  每类一个模型 vs 共享 softmax =====")
    gm = train_softmax(Ztr, y, 10)
    acc_soft = float((pred_logits(gm, Zte).argmax(1) == yte).mean())

    # per-class 一对多：每类一个二分类器，取最大 logit
    scores = np.zeros((len(Zte), 10), np.float32)
    for c in range(10):
        Xt = torch.tensor(Ztr, dtype=torch.float32)
        yb = torch.tensor((y == c).astype(np.float32)).unsqueeze(1)
        m = nn.Linear(Ztr.shape[1], 1)
        nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(m(Xt), yb)
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores[:, c] = m(torch.tensor(Zte, dtype=torch.float32)).squeeze(1).numpy()
    acc_ovr = float((scores.argmax(1) == yte).mean())
    log(f"共享 softmax            {acc_soft:.4f}")
    log(f"每类一个 OvR 模型        {acc_ovr:.4f}   (差 {acc_ovr-acc_soft:+.4f})")
    out["part1"] = dict(softmax=acc_soft, ovr=acc_ovr)

    # ---------------- Part 2: 协作价值 ----------------
    log("\n===== Part 2  全局协作 vs 节点单干（只看本节点那 2 类）=====")
    rng = np.random.RandomState(99)
    rows = []
    for n_per in [20, 50, 100, 300]:
        # 全局：每类抽 n_per 张，10 类共 10*n_per
        gsel = []
        for c in range(10):
            idx = np.where(y == c)[0]
            gsel.append(rng.choice(idx, size=min(n_per, len(idx)), replace=False))
        gsel = np.concatenate(gsel)
        gmodel = train_softmax(Ztr[gsel], y[gsel], 10)

        # 局部：每个节点只有 2 类 × n_per
        loc_acc, glo_acc, n_sub = [], [], 0
        for node in range(20):
            cls = sorted(P["nc"][node])
            mask_tr = np.isin(y, cls) & (P["tr_node"] == node)
            # 该节点每类抽 n_per（不足则全取）
            sel = []
            for c in cls:
                idx = np.where(mask_tr & (y == c))[0]
                sel.append(rng.choice(idx, size=min(n_per, len(idx)), replace=False))
            sel = np.concatenate(sel)
            if len(sel) < 20:
                continue
            remap = {c: i for i, c in enumerate(cls)}
            yloc = np.array([remap[v] for v in y[sel]])
            lmodel = train_softmax(Ztr[sel], yloc, len(cls))

            # 测试子集：同属这 2 类的测试样本
            mte = np.isin(yte, cls)
            if mte.sum() < 10:
                continue
            n_sub += int(mte.sum())
            # 局部模型
            pl = pred_logits(lmodel, Zte[mte]).argmax(1)
            loc_acc.append((np.array(cls)[pl] == yte[mte]).mean())
            # 全局模型，argmax 限制在这 2 类
            pg = pred_logits(gmodel, Zte[mte])[:, cls].argmax(1)
            glo_acc.append((np.array(cls)[pg] == yte[mte]).mean())

        la, ga = float(np.mean(loc_acc)), float(np.mean(glo_acc))
        gacc = float((pred_logits(gmodel, Zte).argmax(1) == yte).mean())
        log(f"n/类={n_per:4d}  单干 {la:.4f}  协作 {ga:.4f}  提升 {ga-la:+.4f}"
            f"   (全局10类准确率 {gacc:.4f})")
        rows.append(dict(n_per=n_per, local=la, coop=ga, gain=ga - la, global_acc=gacc))

    out["part2"] = rows
    with open(os.path.join(ROOT, "results", "results_expert.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n完成 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
