"""exp72: 只让少数代表性节点参与训练 body，各类在空间里能否仍有区分度

背景（作者设想）：φ/body 的更新不必全体节点参与，用贪心选少量代表性节点即可，
目标只是让各个类在坐标空间里有区分度。

CIFAR-100，20 节点 × 5 类（无重叠，故任何少于 20 个节点的子集都覆盖不全 100 类）。
比较四种选点，各自在该子集的合并数据上训练一份共用的 body（384→256→ReLU，
有监督联合训练，这是子集训练能取得的最好情形），再冻结 body、对全部数据重新算
特征，用同一套协议重算判别投影 W 与最近原型路由，读三组数：

  A1_all    全部 100 类的最近原型精度
  A1_seen   参训的 25 类
  A1_unseen 未参训的 75 类（漂移的直接证据）
  Fisher    投影空间的类间散度与类内散度之比，即「各类在空间里的区分度」

选点方法：
  all     20 个节点全参训（对照，覆盖全部类）
  center  坐标空间 k-center 贪心，取最分散的 5 个节点
  hard    取最难分的 5 个节点（自身类到最近异类的距离最小）
  rand    随机 5 个节点，3 个种子取均值
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp32_end2end_all_lda import fit_lda  # noqa: E402
from exp53_c100_end2end_lda import load_c100  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

K = 100
NN = 20
KPN = K // NN          # 每节点 5 类
K_PICK = 5             # 参训节点数
N_TEST = 3000
EPOCHS = 300


def log(*a):
    print(*a, flush=True)


def fisher_ratio(Z, y):
    """投影空间里 trace(S_B)/trace(S_W)。"""
    Z = Z.astype(np.float64)
    m = Z.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = Z[y == c]
        mc = Xc.mean(0)
        sb += len(Xc) * float(((mc - m) ** 2).sum())
        sw += float(((Xc - mc) ** 2).sum())
    return sb / max(sw, 1e-12)


def route_a1(T, yT, P, pcls):
    """最近原型路由：每个查询取最近的异类原型之外的最近原型（按类去重）。"""
    Zq, Zp = T.astype(np.float64), P.astype(np.float64)
    Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                            - 2 * Zq @ Zp.T, 0))
    order = np.argsort(Dm, 1)
    pred = np.zeros(len(T), np.int32)
    for i in range(len(T)):
        seen = set()
        for j in order[i]:
            c = int(pcls[j])
            if c in seen:
                continue
            seen.add(c)
            pred[i] = c
            break
    return pred


def evaluate(H_new, y, T_new, yT, seen_cls):
    mu, W, d = fit_lda(H_new, y, K - 1)
    Z = (H_new - mu) @ W
    Zt = (T_new - mu) @ W
    protos, pcls = [], []
    for c in range(K):
        Xc = Z[y == c]
        if len(Xc) == 0:
            continue
        protos.append(Xc.mean(0))
        pcls.append(c)
    P = np.stack(protos)
    pred = route_a1(Zt, yT, P, np.array(pcls))
    ok = pred == yT
    seen = np.isin(yT, list(seen_cls))
    return dict(d=int(d),
                a1_all=float(ok.mean()),
                a1_seen=float(ok[seen].mean()),
                a1_unseen=float(ok[~seen].mean()),
                fisher=float(fisher_ratio(Z, y)))


def train_body(Hs, ys):
    """在子集合并数据上训练一份共用 body（有监督，head 只是训练用的脚手架）。"""
    nb = 384
    body = nn.Sequential(nn.Linear(nb, 256), nn.ReLU())
    head = nn.Linear(256, len(np.unique(ys)))
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()),
                           lr=5e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Hs.astype(np.float32))
    yt = torch.from_numpy(ys.astype(np.int64))
    for _ in range(EPOCHS):
        opt.zero_grad()
        lossf(head(body(Xt)), yt).backward()
        opt.step()
    body.eval()
    return body


def apply_body(body, H):
    with torch.no_grad():
        out = []
        for i in range(0, len(H), 8192):
            out.append(body(torch.from_numpy(H[i:i + 8192].astype(np.float32))).numpy())
    return np.concatenate(out, 0).astype(np.float32)


def pick_kcenter(Zp_node, k):
    """节点质心先各自算，再做 farthest-first 贪心。"""
    cen = Zp_node
    start = int(np.argmax(((cen - cen.mean(0)) ** 2).sum(1)))
    sel = [start]
    dist = ((cen - cen[start]) ** 2).sum(1)
    while len(sel) < k:
        j = int(np.argmax(dist))
        sel.append(j)
        dist = np.minimum(dist, ((cen - cen[j]) ** 2).sum(1))
    return sorted(sel)


def main():
    H, y, T, yT = load_c100()
    T, yT = T[:N_TEST], yT[:N_TEST]
    keep = np.isin(y, np.arange(K))
    H, y = H[keep], y[keep]
    cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(K), NN))}

    mu, W, d0 = fit_lda(H, y, K - 1)
    log(f"原始特征基线：d={d0}")
    base = evaluate(H, y, T, yT, np.arange(K))
    log(f"  E0 未训练   A1_all {base['a1_all']:.4f}  Fisher {base['fisher']:.2f}")

    # 节点质心（在原始坐标空间里）
    Zp_node = np.stack([((H[np.isin(y, cls_of_node[n])] - mu) @ W).mean(0) for n in range(NN)])

    # 每个节点的难度：其各类到最近异类原型的距离 / 类内散度，取最小（最易混）
    hard = np.zeros(NN)
    for n in range(NN):
        Z = (H[np.isin(y, cls_of_node[n])] - mu) @ W
        protos = np.stack([Z[y[np.isin(y, cls_of_node[n])] == c].mean(0) for c in cls_of_node[n]])
        allp = np.stack([((H[y == c] - mu) @ W).mean(0) for c in range(K)])
        worst = 1e9
        for i, c in enumerate(cls_of_node[n]):
            others = np.delete(allp, c, axis=0)
            dist = np.sqrt(((others - protos[i]) ** 2).sum(1)).min()
            spread = np.sqrt(((Z[y[np.isin(y, cls_of_node[n])] == c] - protos[i]) ** 2).sum(1)).mean()
            worst = min(worst, dist / max(spread, 1e-9))
        hard[n] = worst
    hardest = sorted(np.argsort(hard)[:K_PICK].tolist())

    rng = np.random.default_rng(0)
    plans = {
        "all": list(range(NN)),
        "center": pick_kcenter(Zp_node, K_PICK),
        "hard": hardest,
        "rand1": sorted(rng.choice(NN, K_PICK, replace=False).tolist()),
        "rand2": sorted(rng.choice(NN, K_PICK, replace=False).tolist()),
        "rand3": sorted(rng.choice(NN, K_PICK, replace=False).tolist()),
    }

    res = {"baseline": base, "plans": {}, "hardness": hard.tolist()}
    for tag, nodes in plans.items():
        seen_cls = [c for n in nodes for c in cls_of_node[n]]
        rows = np.isin(y, seen_cls)
        Hs, ys_raw = H[rows], y[rows]
        uniq = {c: i for i, c in enumerate(sorted(set(seen_cls)))}
        ys = np.array([uniq[c] for c in ys_raw], dtype=np.int64)
        t0 = time.time()
        body = train_body(Hs, ys)
        Hn = apply_body(body, H)
        Tn = apply_body(body, T)
        r = evaluate(Hn, y, Tn, yT, seen_cls)
        r["nodes"] = nodes
        r["n_classes_seen"] = len(set(seen_cls))
        res["plans"][tag] = r
        log(f"  {tag:7s} 节点{nodes} 类 {len(set(seen_cls)):3d}  "
            f"A1_all {r['a1_all']:.4f}  seen {r['a1_seen']:.4f}  "
            f"unseen {r['a1_unseen']:.4f}  Fisher {r['fisher']:.2f}  ({time.time()-t0:.0f}s)")

    with open(os.path.join(ROOT, "results", "results_subset_producer.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("写入 results_subset_producer.json")


if __name__ == "__main__":
    main()
