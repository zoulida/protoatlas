"""exp78: 类增长时，判别投影能否沿用旧的一份

第 5 章（原附录 F）主张「W 不能增量修改，只能整份替换」，此前只有定性理由
（维数变、S_B 的秩变、方向重新分配），没有定量依据。本脚本补这一档。

设定：ImageNet-1k 缓存，1000 类，每类 30 张作原型、20 张作查询。
初始类集 50 类，在这一批上拟合 W0（d0 = 49）；之后逐档加入新类，
K = 50 / 100 / 200 / 400 / 600 / 800 / 1000，三档对照：

  A 固定 W0       一直用 50 类时算出的 W0（49 维），新类投进旧子空间
  B 重算但截断     在当档类集上重算 W_K，只取前 49 列（维数与 A 相同）
  C 重算且全维     在当档类集上重算 W_K，d = min(K-1, 384)

A 与 B 的差 = 方向是否随类集更新的贡献；B 与 C 的差 = 维数的贡献。
另报旧类与新类分开的精度，看固定 W 的损失落在哪一侧。

读三组数：A1 最近原型路由精度、margin 类间余量比中位数、旧/新类的分项精度。
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)

from exp32_end2end_all_lda import fit_lda  # noqa: E402

N_PROTO = 30
N_QUERY = 20
NCLS = 1000
K0 = 50
D_FIX = K0 - 1
KS = [50, 100, 200, 400, 600, 800, 1000]
SEEDS = 3
CHUNK = 2000


def log(*a):
    print(*a, flush=True)


def build_all(H, y):
    """为全部 1000 类切出原型集与查询集。"""
    Hp, yp, Hq, yq = [], [], [], []
    for c in range(NCLS):
        X = H[y == c]
        if len(X) < N_PROTO + N_QUERY:
            raise RuntimeError(f"类 {c} 样本不足")
        Hp.append(X[:N_PROTO])
        Hq.append(X[N_PROTO:N_PROTO + N_QUERY])
        yp.append(np.full(N_PROTO, c))
        yq.append(np.full(N_QUERY, c))
    return (np.concatenate(Hp), np.concatenate(yp),
            np.concatenate(Hq), np.concatenate(yq))


def measure(Hp, yp, Hq, yq, m, W, cls):
    """在给定投影下算最近原型精度与类间余量比。cls 与原型行一一对应。"""
    m = m.astype(np.float64)
    Wd = W.astype(np.float64)
    Zp = (Hp.astype(np.float64) - m) @ Wd
    Zq = (Hq.astype(np.float64) - m) @ Wd

    P = np.stack([Zp[yp == c].mean(0) for c in cls])
    spread = float(np.median(
        [np.sqrt(((Zp[yp == c] - Zp[yp == c].mean(0)) ** 2).sum(1)).mean() for c in cls]))

    pn = (P ** 2).sum(1)
    pred = np.empty(len(Zq), np.int64)
    for i in range(0, len(Zq), CHUNK):
        blk = Zq[i:i + CHUNK]
        d2 = (blk ** 2).sum(1)[:, None] + pn[None, :] - 2 * blk @ P.T
        pred[i:i + CHUNK] = np.argmin(d2, 1)
    acc = np.array(cls)[pred] == yq

    Pd = np.sqrt(np.maximum(pn[:, None] + pn[None, :] - 2 * P @ P.T, 0))
    np.fill_diagonal(Pd, np.inf)
    margin = float(np.median(Pd.min(1)) / max(spread, 1e-9))
    return acc, margin


def main():
    H = np.load(os.path.join(CACHE, "feat_in1k.npy")).astype(np.float32)
    y = np.load(os.path.join(CACHE, "in1k_y.npy"))
    log(f"特征 {H.shape}，类数 {len(np.unique(y))}")

    Hp, yp, Hq, yq = build_all(H, y)

    rows = []
    for K in KS:
        acc_ = {"A": [], "B": [], "C": []}
        mg_ = {"A": [], "B": [], "C": []}
        old_ = {"A": [], "B": [], "C": []}
        new_ = {"A": [], "B": [], "C": []}
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 + s)
            order = rng.permutation(NCLS)
            old = order[:K0]
            cls = order[:K]
            mk = np.isin(yp, cls)
            mo = np.isin(yp, old)
            mq = np.isin(yq, cls)

            m0, W0, _ = fit_lda(Hp[mo], yp[mo], D_FIX)
            mK, WK, dK = fit_lda(Hp[mk], yp[mk], min(K - 1, Hp.shape[1]))

            for tag, (mm, WW) in (("A", (m0, W0)),
                                  ("B", (mK, WK[:, :D_FIX])),
                                  ("C", (mK, WK))):
                acc, mg = measure(Hp[mk], yp[mk], Hq[mq], yq[mq], mm, WW, list(cls))
                acc_[tag].append(float(acc.mean()))
                mg_[tag].append(mg)
                if K > K0:
                    is_old = np.isin(yq[mq], old)
                    old_[tag].append(float(acc[is_old].mean()))
                    new_[tag].append(float(acc[~is_old].mean()))

        row = dict(K=K, d_full=int(min(K - 1, Hp.shape[1])))
        for tag, name in (("A", "fix"), ("B", "recalc49"), ("C", "recalc")):
            row[f"a1_{name}"] = float(np.mean(acc_[tag]))
            row[f"a1_{name}_std"] = float(np.std(acc_[tag]))
            row[f"margin_{name}"] = float(np.mean(mg_[tag]))
        for tag, name in (("A", "fix"), ("B", "recalc49"), ("C", "recalc")):
            row[f"old_{name}"] = float(np.mean(old_[tag])) if old_[tag] else float("nan")
            row[f"new_{name}"] = float(np.mean(new_[tag])) if new_[tag] else float("nan")
        rows.append(row)
        log(f"  K={K:5d} d={row['d_full']:4d} | "
            f"A 固定 {row['a1_fix']:.4f} (旧 {row['old_fix']:.4f} 新 {row['new_fix']:.4f}) "
            f"| B 重算49 {row['a1_recalc49']:.4f} | C 全维 {row['a1_recalc']:.4f} "
            f"| margin A {row['margin_fix']:.2f} B {row['margin_recalc49']:.2f} C {row['margin_recalc']:.2f}")

    with open(os.path.join(ROOT, "results", "results_w_update.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log("写入 results_w_update.json")


if __name__ == "__main__":
    main()
