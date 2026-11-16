"""exp44: 类数增长时到底该修哪一层——投影、原型、还是分类器？

背景：exp43 的 B 组（K 10→100）显示 oracle 从 0.848 掉到 0.558（-29 点），
我上一轮把它全部归因为"任务变难"。但那个实验里每类只用 3 个原始样本做原型，
而 exp43 A 组显示"每类 3 样本" vs "每类 1 个类均值" 差 19 点（0.562 vs 0.752）。
所以 -29 点里混了两种东西，必须拆开：

  ① 原型质量（每类样本数 / 用不用均值）
  ② 投影能力（PCA-64 vs LDA，维度是否够）
  ③ 剩下才是"1-NN 本身在高类数下变难"

设计：K × 原型 × 投影 三重扫描，三个因子独立变动。

  投影四档
    PCA-64    现状基线，固定 64 维
    PCA-d     与 LDA 同维（d = min(K-1, 64)），排除维度混淆
    LDA-d     Fisher 判别投影，d = min(K-1, 64)，与 PCA-d 同维
    LDA-max   Fisher 判别投影，d = K-1（不设上限，看判别方向用满的效果）

  原型三档
    per=0     每类 1 个类均值（用该节点全部该类的训练样本）
    per=3     每类 3 个原始样本（冷启动，exp43 B 组口径）
    per=10    每类 10 个原始样本

指标：oracle（全局扫描上界）、1 起点 / 3 起点类命中、跳数。

注意：投影在"被抽中的 K 类的全部训练样本"上拟合，等价于 Federated LDA
一次聚合后的全局 W；查询只用这 K 类的测试样本。
"""
import os
import json
import math
import importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def log(*a):
    print(*a, flush=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e43 = _load("e43", os.path.join(ROOT, "exp43_vnode_who.py"))
e37 = _load("e37", os.path.join(ROOT, "exp37_metric_learn.py"))

run_one = e43.run_one
fit_lda = e37.fit_lda


def fit_pca(H, d):
    mu = H.mean(0)
    X = H - mu
    C = X.T @ X / max(1, len(X) - 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    W = V[:, -d:].astype(np.float32)
    return mu.astype(np.float32), W


def build_protos(Ztr, ytr, K, per):
    """返回 (Z, cls_of)。per=0 用类均值。"""
    if per == 0:
        Z = np.stack([Ztr[ytr == c].mean(0) for c in range(K)]).astype(np.float32)
        return Z, np.arange(K, dtype=np.int32)
    Z = np.concatenate([Ztr[ytr == c][:per] for c in range(K)]).astype(np.float32)
    return Z, np.repeat(np.arange(K), per).astype(np.int32)


def main():
    import pandas as pd

    H = np.load(os.path.join(CACHE, "feat_c100_train.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "c100_train.npz"))["y"].astype(np.int32)
    T = np.load(os.path.join(CACHE, "feat_c100.npy")).astype(np.float32)
    yT = pd.read_parquet(os.path.join(ROOT, "data", "cifar100_test.parquet"))["fine_label"] \
        .to_numpy().astype(np.int32)[:len(T)]
    H = H[:len(y)]
    T = T[:len(yT)]
    log(f"特征 {H.shape}，测试 {T.shape}；每类训练样本 "
        f"{int(np.bincount(y).min())}~{int(np.bincount(y).max())}")

    out = []
    Ks = [10, 20, 50, 100]
    per_list = [0, 3, 10]

    for K in Ks:
        reps = 3 if K < 100 else 1
        acc = {}
        for rep in range(reps):
            rr = np.random.RandomState(100 + rep)
            cls_sel = np.sort(rr.choice(100, size=K, replace=False))
            remap = -np.ones(100, dtype=np.int32)
            remap[cls_sel] = np.arange(K, dtype=np.int32)

            mtr = remap[y] >= 0
            Hk = H[mtr]
            yk = remap[y[mtr]]
            mq = remap[yT] >= 0
            Tk = T[mq]
            yq = remap[yT[mq]]

            d_same = min(K - 1, 64)
            projs = {}
            mu, W = fit_pca(Hk, 64)
            projs["PCA-64"] = (mu, W)
            mu, W = fit_pca(Hk, d_same)
            projs[f"PCA-{d_same}"] = (mu, W)
            mu, W, _ = fit_lda(Hk, yk, d_same)
            projs[f"LDA-{d_same}"] = (mu, W)
            mu, W, _ = fit_lda(Hk, yk, K - 1)
            projs[f"LDA-{W.shape[1]}"] = (mu, W)

            for pname, (mu, W) in projs.items():
                Ztr_all = (Hk - mu) @ W
                Zq = (Tk - mu) @ W
                for per in per_list:
                    Z, cls_of = build_protos(Ztr_all, yk, K, per)
                    M = len(Z)
                    k = int(round(max(8, 6 + 4 * math.log10(max(M, 2) / 100))))
                    r1 = run_one(Zq, yq, Z, cls_of, k=k, n_start=1, seed=3)
                    r3 = run_one(Zq, yq, Z, cls_of, k=k, n_start=3, seed=3)
                    key = (pname, per)
                    acc.setdefault(key, []).append(
                        dict(oracle=r1["oracle"], hit1=r1["hit"], hit3=r3["hit"],
                             hops=r1["hops"], M=M, dim=int(W.shape[1])))

        # 汇总
        rows = []
        for (pname, per), lst in acc.items():
            avg = {kk: float(np.mean([a[kk] for a in lst])) for kk in lst[0]}
            rows.append(dict(K=K, proj=pname, per=per, n_rep=len(lst), **avg))
        rows.sort(key=lambda r: (r["per"], r["proj"]))
        out += rows

        log("")
        log(f"=== K={K}（{reps} 次重复平均，每类训练样本 {int(np.bincount(yk).min())}~"
            f"{int(np.bincount(yk).max())}）===")
        log(f"{'投影':>10}{'维':>4}{'原型':>7}{'M':>6}{'oracle':>9}{'1起点':>9}{'3起点':>9}{'跳数':>7}")
        for r in rows:
            pname = r["proj"]
            ptag = "类均值" if r["per"] == 0 else f"{r['per']}样本"
            log(f"{pname:>10}{r['dim']:>4}{ptag:>7}{r['M']:>6}"
                f"{r['oracle']:>9.4f}{r['hit1']:>9.4f}{r['hit3']:>9.4f}{r['hops']:>7.2f}")

    with open(os.path.join(ROOT, "results", "results_class_growth.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("")
    log("已存 results_class_growth.json")

    # ---------------- 归因小结 ----------------
    log("")
    log("=== 归因：K=100 相对 K=10 的 oracle 损失怎么分摊 ===")
    def get(K, pname, per):
        for r in out:
            if r["K"] == K and r["proj"].startswith(pname) and r["per"] == per:
                return r
        return None

    for pname in ["PCA-64", "LDA-"]:
        for per in [3, 0]:
            a, b = get(10, pname, per), get(100, pname, per)
            if a and b:
                log(f"  {pname:<9} per={per:<2}: oracle {a['oracle']:.4f} → "
                    f"{b['oracle']:.4f}  ({b['oracle'] - a['oracle']:+.4f})")

    log("")
    log("=== 拆解（K=100，逐项替换）===")
    base = get(100, "PCA-64", 3)
    d_only = get(100, "PCA-64", 0)
    lda_only = get(100, "LDA-99", 3) or get(100, "LDA-64", 3)
    both = get(100, "LDA-99", 0) or get(100, "LDA-64", 0)
    if base and d_only:
        log(f"  基线 PCA-64 + 3样本          oracle={base['oracle']:.4f}  hit3={base['hit3']:.4f}")
        log(f"  + 原型换成类均值              oracle={d_only['oracle']:.4f}  "
            f"({d_only['oracle'] - base['oracle']:+.4f})  hit3={d_only['hit3']:.4f}")
    if base and lda_only:
        log(f"  + 投影换成 LDA（仍 3 样本）    oracle={lda_only['oracle']:.4f}  "
            f"({lda_only['oracle'] - base['oracle']:+.4f})  hit3={lda_only['hit3']:.4f}")
    if base and both:
        log(f"  + 两者都换                    oracle={both['oracle']:.4f}  "
            f"({both['oracle'] - base['oracle']:+.4f})  hit3={both['hit3']:.4f}")


if __name__ == "__main__":
    main()
