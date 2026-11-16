"""exp87: 簇内共同训练 body 能不能加快收敛（纯 numpy 批量版）

问题：body 在一组坐标邻近的节点之间定期平均，相比各节点各自训练，
能否在相同的本地步数下更快达到同一精度，或达到更高精度。

与 exp81/exp82 的差别：
  - 训练改为全部节点批量并行（每个节点一份参数，批量 matmul），
    因而可以每隔若干步记一次曲线，而不是只报终点；
  - 同步档不再每轮重置判别头，本地步数与本地档完全相同，
    两档的差别只剩「body 要不要平均」这一项；
  - 记通信量，用于判断收敛更快时是否值得。

两档：
  L   本地训练，body 与 head 都只用自己的数据，不交换
  S   本地训练，每 interval 步把 body（W1,b1）在 8 个坐标最近节点间取平均
      （含自身，等权），head 始终本地

读三个量：
  acc(step)  各节点本地测试精度的均值随本地步数的曲线
  T_tgt      达到给定精度所需的本地步数
  bytes      同步档额外付出的通信量

用法：
  python exp87_body_sync_speed.py 1k [--steps 300] [--intervals 10,50] [--seeds 3]
  python exp87_body_sync_speed.py c100
"""
import os
import sys
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

D_IN, HID, LR, WD = 384, 256, 5e-3, 1e-4
WD_BODY, WD_HEAD = WD, WD
EVAL_POINTS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 300]


def log(*a):
    print(*a, flush=True)


def init_state(N, C, rng):
    """每个节点一份参数，输出维 C 相同（本实验各节点类数相等）"""
    return dict(
        W1=(rng.randn(N, D_IN, HID) * np.sqrt(2.0 / D_IN)).astype(np.float32),
        b1=np.zeros((N, 1, HID), np.float32),
        W2=(rng.randn(N, HID, C) * np.sqrt(1.0 / HID)).astype(np.float32),
        b2=np.zeros((N, 1, C), np.float32))


def forward(st, X):
    F = np.maximum(X @ st["W1"] + st["b1"], 0)
    return F, F @ st["W2"] + st["b2"]


def grads(st, X, y, F, logit):
    L = logit - logit.max(2, keepdims=True)
    e = np.exp(L)
    P = e / e.sum(2, keepdims=True)
    S = y.shape[1]
    g = P
    np.put_along_axis(g, y[:, :, None],
                      np.take_along_axis(g, y[:, :, None], 2) - 1.0, 2)
    g /= S
    gW2 = F.transpose(0, 2, 1) @ g + WD_HEAD * st["W2"]
    gb2 = g.sum(1, keepdims=True)
    dF = (g @ st["W2"].transpose(0, 2, 1)) * (F > 0)
    gW1 = X.transpose(0, 2, 1) @ dF + WD_BODY * st["W1"]
    gb1 = dF.sum(1, keepdims=True)
    return dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2)


class Adam:
    def __init__(self, st):
        self.m = {k: np.zeros_like(v) for k, v in st.items()}
        self.v = {k: np.zeros_like(v) for k, v in st.items()}
        self.t = 0

    def step(self, st, gr):
        self.t += 1
        c1 = 1 - 0.9 ** self.t
        c2 = 1 - 0.999 ** self.t
        for k, g in gr.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            st[k] -= (LR * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)
                      ).astype(np.float32)


def accuracy(st, X, y):
    _, logit = forward(st, X)
    return float((logit.argmax(2) == y).mean())


def load_1k(per_class=25):
    """ImageNet-1k：1000 类，50 节点 × 20 类，每类 per_class 训、其余测"""
    Ht = np.load(os.path.join(CACHE, "feat_in1k.npy"))
    yt = np.load(os.path.join(CACHE, "in1k_y.npy")).astype(np.int64)
    K, N, NTR, NTE = 1000, 50, per_class, 25
    chunks = [list(range(i * K // N, (i + 1) * K // N)) for i in range(N)]
    Xtr, ytr, Xte, yte, protos = [], [], [], [], []
    for cols in chunks:
        a, b = [], []
        for j, c in enumerate(cols):
            rows = np.where(yt == c)[0]
            a.append(Ht[rows[:NTR]])
            b.append(Ht[rows[NTE:]])
            protos.append(Ht[rows[:NTR]].mean(0))
        Xtr.append(np.concatenate(a))
        Xte.append(np.concatenate(b))
        ytr.append(np.repeat(np.arange(len(cols)), NTR))
        yte.append(np.repeat(np.arange(len(cols)), NTE))
    return (np.stack(Xtr), np.stack(ytr), np.stack(Xte), np.stack(yte),
            np.stack(protos))


def load_c100():
    """CIFAR-100：200 节点 × 2 类，每类 125 训 100 测，每类 4 副本"""
    from exp1_route import make_shard2
    Htr = np.load(os.path.join(CACHE, "feat_c100_train.npy"))
    ytr_all = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int64)
    n = min(len(Htr), len(ytr_all))
    Htr, ytr_all = Htr[:n], ytr_all[:n]
    nc, ni = make_shard2(ytr_all, 200, 100, 2, seed=0)
    Tte = np.load(os.path.join(CACHE, "feat_c100_test10k.npy"))
    yte_all = np.load(os.path.join(CACHE, "c100_test10k.npz"))["y"].astype(np.int64)
    rng = np.random.RandomState(1)
    Xtr, ytr, Xte, yte, protos = [], [], [], [], []
    for nn in range(200):
        cols = sorted(nc[nn])
        a, b, t, u = [], [], [], []
        for j, c in enumerate(cols):
            sel = ni[nn][ytr_all[ni[nn]] == c]
            take = rng.choice(sel, size=min(125, len(sel)), replace=False)
            a.append(Htr[take])
            t.append(np.full(len(take), j, np.int64))
            protos.append(Htr[take].mean(0))
            sel_t = np.where(yte_all == c)[0]
            take_t = rng.choice(sel_t, size=min(100, len(sel_t)), replace=False)
            b.append(Tte[take_t])
            u.append(np.full(len(take_t), j, np.int64))
        Xtr.append(np.concatenate(a))
        ytr.append(np.concatenate(t))
        Xte.append(np.concatenate(b))
        yte.append(np.concatenate(u))
    return (np.stack(Xtr), np.stack(ytr), np.stack(Xte), np.stack(yte),
            np.stack(protos))


def neighbors(node_z, k=8):
    d = ((node_z[:, None, :] - node_z[None, :, :]) ** 2).sum(-1)
    return [list(np.argsort(d[n])[:k]) for n in range(len(node_z))]


def mix_matrix(nbr, N):
    """邻居平均矩阵，A[i,j] = 1/k 当 j 属于 i 的邻居集合（含自身）"""
    A = np.zeros((N, N), np.float32)
    for i, js in enumerate(nbr):
        A[i, js] = 1.0 / len(js)
    return A


def run_arm(Xtr, ytr, Xte, yte, A, seed, steps, interval, evals, stop=None):
    N, C = ytr.shape[0], ytr.max() + 1
    rng = np.random.RandomState(seed)
    st = init_state(N, C, rng)
    opt = Adam(st)
    curve, rounds = [], 0
    ev = set(evals)
    for s in range(1, steps + 1):
        # 同步落在每轮开头，测量落在每轮末尾，故读数反映的是「同步后再训练
        # interval-1 步」的状态，与本地档在同一步数上可比
        if (interval and s > 1 and (s - 1) % interval == 0
                and (stop is None or s <= stop)):
            W1 = st["W1"].reshape(N, -1)
            b1 = st["b1"].reshape(N, -1)
            st["W1"] = (A @ W1).reshape(st["W1"].shape)
            st["b1"] = (A @ b1).reshape(st["b1"].shape)
            rounds += 1
        F, logit = forward(st, Xtr)
        opt.step(st, grads(st, Xtr, ytr, F, logit))
        if s in ev:
            curve.append((s, accuracy(st, Xte, yte)))
    return curve, rounds, accuracy(st, Xte, yte)


def steps_to(curve, thr):
    for s, a in curve:
        if a >= thr:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", choices=["1k", "c100"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--intervals", default="10,50")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="把权重衰减改为该值（用于检验同步的增益是否只是正则化）")
    ap.add_argument("--wd-body-only", action="store_true",
                    help="只对 body 加权衰减，判别头保持默认，排除头正则的混淆")
    ap.add_argument("--early", type=int, default=0,
                    help="补一档：每 10 步同步、但只在前 N 步内同步，之后纯本地")
    ap.add_argument("--ablation", action="store_true",
                    help="补一档：同步对象改为 8 个随机节点，检验邻近性是否必要")
    a = ap.parse_args()
    if a.wd:
        global WD_BODY, WD_HEAD
        WD_BODY = a.wd
        if not a.wd_body_only:
            WD_HEAD = a.wd
    t0 = time.time()

    Xtr, ytr, T, yT, protos = (load_1k(a.per_class) if a.data == "1k"
                              else load_c100())
    per = len(protos) // len(Xtr)
    node_z = np.stack([protos[i * per:(i + 1) * per].mean(0) for i in range(len(Xtr))])
    nbr = neighbors(node_z, 8)
    log(f"{a.data}: 节点 {Xtr.shape[0]}  每节点训练 {Xtr.shape[1]} 测试 {T.shape[1]} "
        f"类数 {int(ytr.max())+1}")

    intervals = [int(x) for x in a.intervals.split(",") if x]
    res = {"data": a.data, "n_nodes": int(Xtr.shape[0]),
           "cls_per_node": int(ytr.max() + 1), "steps": a.steps,
           "train_per_node": int(Xtr.shape[1]),
           "eval_points": EVAL_POINTS, "seeds": a.seeds,
           "body_bytes": int(D_IN * HID * 4 + HID * 4), "arms": {}}

    A = mix_matrix(nbr, len(Xtr))
    evals = [e for e in EVAL_POINTS if e <= a.steps] + [a.steps]
    evals = sorted(set(evals))
    arms = [("L", 0, A, None)] + [(f"S{iv}", iv, A, None) for iv in intervals]
    if a.early:
        arms.append((f"E{a.early}", 10, A, a.early))
    if a.ablation:
        ra = np.random.RandomState(12345)
        nbr_r = [list(ra.choice(len(Xtr), 8, replace=False)) for _ in range(len(Xtr))]
        arms.append(("R10", 10, mix_matrix(nbr_r, len(Xtr)), None))
    for tag, iv, M, stp in arms:
        curves, finals, rounds = [], [], []
        for sd in range(a.seeds):
            c, r, f = run_arm(Xtr, ytr, T, yT, M, sd, a.steps, iv, evals, stp)
            curves.append([v for _, v in c])
            finals.append(f)
            rounds.append(r)
            log(f"  {tag} seed{sd} 终点 {f:.4f}  第 {a.steps} 步  用时 {time.time()-t0:.0f}s")
        means = np.mean(np.array(curves), 0)
        stds = np.std(np.array(curves), 0)
        steps_axis = evals[:len(means)]
        res["arms"][tag] = dict(
            interval=iv, final=float(np.mean(finals)), final_std=float(np.std(finals)),
            rounds=int(rounds[0]) if rounds else 0,
            curve_mean=[round(float(x), 4) for x in means],
            curve_std=[round(float(x), 4) for x in stds],
            steps=steps_axis)
        log(f"{tag:5s} 终点 {np.mean(finals):.4f}±{np.std(finals):.4f}  "
            f"同步 {res['arms'][tag]['rounds']} 轮  "
            f"曲线 {means[::3].round(3).tolist()}")

    # 达到同一精度所需步数
    base = res["arms"]["L"]
    for tag, r in res["arms"].items():
        r["steps_to"] = {}
        for q in (0.95, 0.98, 0.99):
            thr = q * max(base["final"], r["final"])
            r["steps_to"][f"{int(q*100)}%"] = steps_to(list(zip(r["steps"], r["curve_mean"])), thr)
        # 通信量：每轮每节点向 8 个邻居发送、收 8 份
        r["bytes_per_node"] = r["rounds"] * 8 * 2 * res["body_bytes"]
        r["bytes_all"] = r["bytes_per_node"] * res["n_nodes"]

    suf = (f"_early{a.early}" if a.early else "") + ("_rand" if a.ablation else "")\
          + (f"_wd{a.wd:g}" if a.wd else "") + ("_body" if a.wd_body_only else "")
    out = os.path.join(ROOT,
                       f"results_body_sync_speed_{a.data}_pc{a.per_class}{suf}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log(f"写入 {os.path.basename(out)}  总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
