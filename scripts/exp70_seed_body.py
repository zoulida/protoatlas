"""exp70: 补种子——body 的增益到底有多稳

exp69 与 exp67b 在 ImageNet-100 的 N=8 档上给出相反的结论（端到端增益
+0.72 对 +1.80），说明单次运行不足以区分几个点以内的差别。本脚本对
N=1 与 N=8 各跑 5 个种子，报本地判别与端到端的均值 ± 标准差。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp61_body_head import fit_lda  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_NODES = 5


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


def infer(body, head, X, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(head(body(torch.from_numpy(X[i:i + bs]))).numpy())
    return np.concatenate(out, 0)


def main():
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    te_idx = np.concatenate([np.where(yt == c)[0][25:50] for c in range(100)])
    T, yT = Ht[te_idx], yt[te_idx]

    res = {}
    for N in (1, 8):
        f = os.path.join(CACHE, f"feat_in1k100_aug{N}.npy")
        H = np.load(f).astype(np.float32)
        y = np.load(os.path.join(CACHE, f"in1k100_aug{N}_y.npy")).astype(np.int64)
        K = 100
        cls_of_node = {n: list(a) for n, a in enumerate(np.array_split(np.arange(K), N_NODES))}
        mu, W, d_eff = fit_lda(H, y, K - 1)
        protos, pcls, pnode = [], [], []
        for n in range(N_NODES):
            for c in cls_of_node[n]:
                protos.append(H[y == c].mean(0))
                pcls.append(c)
                pnode.append(n)
        P = np.stack(protos).astype(np.float32)
        pcls, pnode = np.array(pcls), np.array(pnode)
        Zq, Zp = (T - mu) @ W, (P - mu) @ W
        Dm = np.sqrt(np.maximum((Zq ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :]
                                - 2 * Zq @ Zp.T, 0))
        top_node = np.array([int(pnode[j]) for j in np.argmin(Dm, 1)])
        top_cls = np.array([int(pcls[j]) for j in np.argmin(Dm, 1)])
        route = float(np.mean(top_cls == yT))

        Hn, yn, rows_of = {}, {}, {}
        for n in range(N_NODES):
            m = np.isin(y, cls_of_node[n])
            Hn[n] = H[m]
            yn[n] = np.array([cls_of_node[n].index(c) for c in y[m]], dtype=np.int64)
            rows_of[n] = np.isin(yT, cls_of_node[n])

        log(f"\n=== N={N}，每节点 {len(H)//N_NODES} 张，路由 {route:.4f} ===")
        out = {"N": N, "route": route, "runs": []}
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rec = {}
            for cfg, hid in (("E0", None), ("E1", 256)):
                bodies, heads = {}, {}
                for n in range(N_NODES):
                    bodies[n], heads[n] = build(hid, K // N_NODES)
                    train_join(Hn[n], yn[n], bodies[n], heads[n], 300)
                loc_r = loc_t = d1_r = 0
                for n in range(N_NODES):
                    m = rows_of[n]
                    if m.sum():
                        pr = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T[m]).argmax(1)]
                        loc_r += int((pr == yT[m]).sum())
                        loc_t += int(m.sum())
                    m2 = top_node == n
                    if m2.sum():
                        pr2 = np.array(cls_of_node[n])[infer(bodies[n], heads[n], T[m2]).argmax(1)]
                        d1_r += int((pr2 == yT[m2]).sum())
                rec[cfg] = {"local": loc_r / loc_t, "D1": d1_r / len(T)}
            out["runs"].append(rec)
            log(f"  seed {seed}: 本地 {rec['E0']['local']:.4f}→{rec['E1']['local']:.4f}"
                f"（{(rec['E1']['local']-rec['E0']['local'])*100:+.2f}）  "
                f"端到端 {rec['E0']['D1']:.4f}→{rec['E1']['D1']:.4f}"
                f"（{(rec['E1']['D1']-rec['E0']['D1'])*100:+.2f}）")

        for key in ("local", "D1"):
            e0 = np.array([r["E0"][key] for r in out["runs"]])
            e1 = np.array([r["E1"][key] for r in out["runs"]])
            g = (e1 - e0) * 100
            out[key] = {"E0_mean": float(e0.mean()), "E1_mean": float(e1.mean()),
                        "gain_mean": float(g.mean()), "gain_std": float(g.std(ddof=1))}
            log(f"  {key}: 增益 {g.mean():+.2f} ± {g.std(ddof=1):.2f} 点"
                f"（E0 {e0.mean():.4f} → E1 {e1.mean():.4f}）")
        res[str(N)] = out

    with open(os.path.join(ROOT, "results", "results_seed_body.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    log("\n写入 results_seed_body.json")


if __name__ == "__main__":
    main()
