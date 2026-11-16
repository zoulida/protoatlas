"""exp13: 两件事
① 导出真实查询的 top-8 原型列表，看清"按类去重"在做什么
② 实测节点拒识：路由送错的请求，节点能不能挡住
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def d2mat(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def build():
    import exp1_route as E
    from exp12_proto_route import build_split

    nc, ni, tr_sel, tr_node, te_sel, ytr, yte, C = build_split()
    H = np.load(os.path.join(CACHE, "feat_train.npy")).astype(np.float32)
    T = np.load(os.path.join(CACHE, "feat_test.npy")).astype(np.float32)
    y_tr, y_te = ytr[tr_sel], yte[te_sel]

    protos, pcls, pnode = [], [], []
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            if rows.sum() == 0:
                continue
            protos.append(H[rows].mean(0))
            pcls.append(k)
            pnode.append(n)
    P = np.stack(protos).astype(np.float32)
    return (H, T, P, np.array(pcls), np.array(pnode),
            y_tr, y_te, tr_node, nc, C)


def main():
    H, T, P, pcls, pnode, y_tr, y_te, tr_node, nc, C = build()
    NAME = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]

    mu = H.mean(0, keepdims=True)
    Cc = (H - mu).T @ (H - mu) / (len(H) - 1)
    w, V = np.linalg.eigh(Cc.astype(np.float64))
    W = V[:, -16:].astype(np.float32)

    Zq = (T - mu) @ W
    Zp = (P - mu) @ W
    D = np.sqrt(d2mat(Zq, Zp))
    order = np.argsort(D, axis=1)

    # ---------- ① 案例导出 ----------
    log("=" * 78)
    log("① 真实查询的 top-8 原型（PCA-16 坐标，距离越小越近）")
    log("=" * 78)

    def show(i, tag):
        log(f"\n[{tag}] 查询 i={i}，真实类 = {NAME[y_te[i]]}")
        log(f"  {'名次':<5}{'原型id':<9}{'类':<7}{'宿主节点':<10}{'距离':<10}{'去重后?':<8}")
        seen = set()
        for r, j in enumerate(order[i, :8]):
            c = pcls[j]
            first = c not in seen
            seen.add(c)
            log(f"  {r+1:<5}{j:<9}{NAME[c]:<6}{pnode[j]:<10}{D[i,j]:<10.4f}"
                f"{'✓ 取' if first else '✗ 跳过(同类已有)'}")
        # 去重结果
        seen, picked = set(), []
        for j in order[i]:
            c = pcls[j]
            if c not in seen:
                seen.add(c)
                picked.append((c, pnode[j], D[i, j]))
            if len(picked) == 3:
                break
        log(f"  → 去重后候选：{[f'{NAME[c]}(节点{n},{d:.3f})' for c, n, d in picked]}")

    # 找一个"top1 正确"的和一个"top1 错误"的
    ok = [i for i in range(len(y_te)) if pcls[order[i, 0]] == y_te[i]]
    bad = [i for i in range(len(y_te)) if pcls[order[i, 0]] != y_te[i]]
    show(ok[0], "情形A：top-1 直接命中，去重只是补充")
    show(bad[0], "情形B：top-1 错了，但正确类仍在去重名单里")
    if len(bad) > 5:
        show(bad[5], "情形C：另一个 top-1 错误样本")

    # ---------- ② 节点拒识 ----------
    log("\n" + "=" * 78)
    log("② 节点拒识：路由送错了，节点能不能挡住？")
    log("=" * 78)

    # 每个节点每类的"自己的半径"：本地训练样本到该类原型的距离分位数
    radii = {}
    for n in range(C["n_nodes"]):
        m = tr_node == n
        for k in nc[n]:
            rows = m & (y_tr == k)
            if rows.sum() < 20:
                continue
            pr = H[rows].mean(0)
            pr16 = (pr - mu[0]) @ W
            dloc = np.sqrt(((((H[rows] - mu[0]) @ W) - pr16) ** 2).sum(1))
            radii[(n, k)] = (dloc, float(np.percentile(dloc, 95)),
                             float(np.median(dloc)))

    # 全局固定阈值（对比：不分节点尺度）
    allr = np.concatenate([v[0] for v in radii.values()])
    g95 = float(np.percentile(allr, 95))
    g99 = float(np.percentile(allr, 99))

    def node_accepts(n, k, z, q, mode):
        if (n, k) not in radii:
            return False, np.inf
        dloc, r, med = radii[(n, k)]
        j = next(j for j in range(len(P)) if pnode[j] == n and pcls[j] == k)
        d = float(np.sqrt(((z - Zp[j]) ** 2).sum()))
        if mode == "local":
            thr = np.percentile(dloc, q)
        else:
            thr = np.percentile(allr, q)
        return d <= thr, d / max(med, 1e-9)

    res = {}
    for mode in ["local", "global"]:
        for q in [90, 95, 99]:
            # 端到端：按类去重 top-3 依次询问，第一个受理的答
            cover = 0           # 正确类最终被受理
            answered_right = 0
            answered_wrong = 0  # 危险：受理了错的类
            all_rej = 0
            first_right = 0     # 第一次询问就受理且正确
            first_wrong_but_rej = 0  # 第一次送错但被挡住
            for i in range(len(y_te)):
                z = Zq[i]
                seen, cand = set(), []
                for j in order[i]:
                    c = pcls[j]
                    if c not in seen:
                        seen.add(c)
                        cand.append((c, pnode[j]))
                    if len(cand) == 3:
                        break
                got = None
                for pos, (c, n) in enumerate(cand):
                    okk, ratio = node_accepts(n, c, z, q, mode)
                    if pos == 0 and okk and c == y_te[i]:
                        first_right += 1
                    if pos == 0 and (not okk) and c != y_te[i]:
                        first_wrong_but_rej += 1
                    if okk:
                        got = c
                        break
                if got is None:
                    all_rej += 1
                elif got == y_te[i]:
                    cover += 1
                    answered_right += 1
                else:
                    answered_wrong += 1
            N = len(y_te)
            key = f"{mode}-p{q}"
            res[key] = dict(
                coverage=cover / N, correct=answered_right / N,
                dangerous_wrong=answered_wrong / N, all_rejected=all_rej / N,
                first_hop_right=first_right / N,
                first_hop_wrong_blocked=first_wrong_but_rej / N,
            )
            log(f"\n[{key}]")
            log(f"  端到端答对（正确类被受理）    {cover/N:.4f}")
            log(f"  危险错误（受理了错的类）      {answered_wrong/N:.4f}")
            log(f"  全部候选拒识（明确说无人可答）{all_rej/N:.4f}")
            log(f"  └ 第一跳就送对且受理          {first_right/N:.4f}")
            log(f"  └ 第一跳送错但被拒识挡住      {first_wrong_but_rej/N:.4f}")

    # 阈值参考
    med_ratio = [v[2] for v in radii.values()]
    r95 = [v[1] for v in radii.values()]
    log(f"\n阈值参考：节点中位半径 {np.mean(med_ratio):.3f}，"
        f"p95 半径 {np.mean(r95):.3f}（40 个原型的均值）")
    log(f"全局 p95 阈值（不做尺度归一）{g95:.3f} / p99 {g99:.3f}")

    with open(os.path.join(ROOT, "results", "results_dedup_reject.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
