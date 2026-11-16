"""exp33: 类别数 K 的上限在哪里 —— 真实类集 K=10/20/100 三点 + 每类样本量影响

问题：类别数在应用场景上应远多于实验设定，极限情况下上限该设多少？

本实验回答两件事：
  ① K 从 10 → 20 → 100（全部是**真实语义类**，非合成），达到同一路由召回
     需要多少维 d？给出 d*(K) 的三个真点，用于外推。
  ② 每类样本数从 500 降到 25（大 K 场景必然发生），会额外损失多少？

口径与 exp25 一致：原型 = 每类训练特征均值，召回@k = 最近 k 个原型里是否
包含真实类（全局扫描，不含分布式损失）。全维 d=384 是该口径的天花板。

三类点来源：
  K=10  CIFAR-10   feat_train(12000,384) / feat_test(5000,384)
  K=20  CIFAR-100 粗标签 feat_c100_train(50000,384) / feat_c100(3000,384)
  K=100 CIFAR-100 细标签 同上
统一控制：每类训练 500 样本（K<100 时截断），测试集均衡不重采样。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def dist(A, B):
    return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                              - 2.0 * A @ B.T, 0))


def recall_at(D, yq, cls_of_proto, ks=(1, 3, 10)):
    yq = yq[:len(D)]
    order = np.argsort(D, 1)
    out = {}
    for k in ks:
        got = cls_of_proto[order[:, :k]]
        out[k] = float(np.mean([(got[i] == yq[i]).any() for i in range(len(yq))]))
    return out


def load_sets():
    """返回 [(name, H, yh, T, yT), ...]

    ⚠ feat_train/feat_test 的缓存是按 exp1 的 20 节点划分顺序抽的，
      标签必须用 build_split() 的 tr_sel / te_sel 还原，不能直接读 parquet 列。
    """
    import pandas as pd
    from exp12_proto_route import build_split
    sets = []

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H10 = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T10 = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    assert len(H10) == len(tr_sel) and len(T10) == len(te_sel)
    sets.append(("K=10", H10, ytr[tr_sel].astype(np.int32), T10, yte[te_sel].astype(np.int32)))

    Hc = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    zc = np.load(os.path.join(CACHE, "c100_train.npz"))
    Tc = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    dte = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))
    sets.append(("K=20", Hc, zc["coarse"].astype(np.int32), Tc, dte["coarse_label"].to_numpy().astype(np.int32)))
    sets.append(("K=100", Hc, zc["y"].astype(np.int32), Tc, dte["fine_label"].to_numpy().astype(np.int32)))
    return sets


def subsample_per_class(H, y, per):
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        if len(ic) > per:
            ic = ic[:per]
        idx.append(ic)
    idx = np.concatenate(idx)
    return H[idx], y[idx]


def main():
    sets = load_sets()
    dims = [16, 32, 64, 128, 192, 256, 384]

    log("=" * 96)
    log("Part A  K 扩展（真实类集，每类训练 500 样本，测试集全部均衡）")
    log("=" * 96)
    log(f"  {'K':>6}{'训练数':>8}{'测试数':>7}{'类内距':>9}{'类间距':>9}{'比':>7}"
        f"{'@1(384)':>9}{'@3(384)':>9}{'@10(384)':>10}")
    log("-" * 96)

    partA = []
    packed = {}
    for name, H, y, T, yT in sets:
        H2, y2 = subsample_per_class(H, y, 500)
        K = len(np.unique(y2))
        mu = H2.mean(0, keepdims=True)
        Cc = (H2 - mu).T @ (H2 - mu) / (len(H2) - 1)
        w, V = np.linalg.eigh(Cc.astype(np.float64))
        V = V[:, ::-1].astype(np.float32)

        protos = np.stack([H2[y2 == c].mean(0) for c in np.unique(y2)]).astype(np.float32)
        cls_p = np.unique(y2)
        # 类内 / 类间距离（全维）
        d_in = np.mean([np.linalg.norm(H2[i] - protos[np.where(cls_p == y2[i])[0][0]])
                        for i in range(0, len(H2), 5)])
        dp = dist(protos, protos)
        np.fill_diagonal(dp, np.inf)
        d_out = float(dp.min(1).mean())

        D384 = dist(T, protos)
        r384 = recall_at(D384, yT, cls_p)
        log(f"  {K:>6}{len(H2):>8}{len(T):>7}{d_in:>9.2f}{d_out:>9.2f}{d_out/d_in:>7.3f}"
            f"{r384[1]:>9.4f}{r384[3]:>9.4f}{r384[10]:>10.4f}")

        rows = []
        for dim in dims:
            W = V[:, :dim]
            Zp = (protos - mu) @ W
            Zq = (T - mu) @ W
            r = recall_at(dist(Zq, Zp), yT, cls_p)
            rows.append(dict(d=dim, r1=r[1], r3=r[3], r10=r[10]))
        partA.append(dict(K=K, n_train=int(len(H2)), n_test=int(len(T)),
                          d_in=float(d_in), d_out=float(d_out), ratio=float(d_out / d_in), scans=rows))
        packed[name] = (V, mu, protos, cls_p, H2, y2, T, yT, K)

    # 维度扫描表
    log("")
    log("  召回@3 随维度变化：")
    head = "  " + f"{'d':>5}" + "".join(f"{p['K']:>10}" for p in partA)
    log(head)
    log("  " + "-" * (len(head) - 2))
    for i, dim in enumerate(dims):
        line = f"  {dim:>5}" + "".join(f"{p['scans'][i]['r3']:>10.4f}" for p in partA)
        log(line)

    # d*(K)：达到 0.95 召回@3 的最小维度
    log("")
    log("  达到 召回@3 = 0.95 所需最小维度：")
    dstar = {}
    for p in partA:
        hit = [r["d"] for r in p["scans"] if r["r3"] >= 0.95]
        dstar[p["K"]] = hit[0] if hit else None
        log(f"    K={p['K']:<4} -> d* = {hit[0] if hit else '>384（达不到）'}")

    # log-log 斜率（用能达到的点）
    xs = [np.log(p["K"]) for p in partA if dstar[p["K"]]]
    ys = [np.log(dstar[p["K"]]) for p in partA if dstar[p["K"]]]
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    log(f"    log-log 拟合斜率 d* ∝ K^{slope:.3f}  →  K 涨 10 倍，d* 涨 {10**slope:.2f} 倍")

    log("")
    log("=" * 96)
    log("Part B  每类样本量缩减的影响（K=100 固定）")
    log("=" * 96)
    V, mu, protos, cls_p, H2, y2, T, yT, K = packed["K=100"]
    log(f"  {'每类样本':>9}{'@1(d=64)':>11}{'@3(d=64)':>11}{'@3(d=384)':>12}{'@10(d=384)':>13}")
    log("-" * 96)
    partB = []
    for per in [500, 250, 100, 50, 25, 10]:
        Hs, ys = subsample_per_class(H2, y2, per)
        ps = np.stack([Hs[ys == c].mean(0) for c in cls_p]).astype(np.float32)
        Zp64 = (ps - mu) @ V[:, :64]
        Zq64 = (T - mu) @ V[:, :64]
        r64 = recall_at(dist(Zq64, Zp64), yT, cls_p)
        r384 = recall_at(dist(T, ps), yT, cls_p)
        log(f"  {per:>9}{r64[1]:>11.4f}{r64[3]:>11.4f}{r384[3]:>12.4f}{r384[10]:>13.4f}")
        partB.append(dict(per=per, r1_64=r64[1], r3_64=r64[3], r3_384=r384[3], r10_384=r384[10]))

    out = dict(partA=partA, partB=partB, dstar=dstar, slope=slope)
    with open(os.path.join(ROOT, "results", "results_K_scale.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("\n  判读：d* 的 log-log 斜率 > 0.5 说明维度需求跑得比类数快，"
        "大 K 场景不能靠固定 16/64 维撑住。")


if __name__ == "__main__":
    main()
