"""verify_w: 确认 W 的切片写法（用真实特征），避免取到方差最小的 16 维"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = np.load(os.path.join(ROOT, "cache", "feat_train.npy")).astype(np.float32)
n, d = H.shape

mu = H.mean(0, keepdims=True)
Hc = H - mu
C = (Hc.T @ Hc) / (n - 1)
w, V = np.linalg.eigh(C.astype(np.float64))

print(f"特征矩阵 {H.shape}")
print(f"eigh 返回：前 3 个特征值 = {np.round(w[:3], 4)}   后 3 个 = {np.round(w[-3:], 2)}")
print("→ eigh 是升序，最大值在最后一列，所以最大的 16 个 = V[:, -16:]\n")

W_ok = V[:, -16:]                    # 正确
W_rev = V[:, ::-1][:, :16]           # 正确（先反转，列顺序相反但子空间相同）
W_bad = V[:, ::-1][:, -16:]          # 错误（反转后又取末尾 = 方差最小）


def var_frac(W):
    Z = Hc @ W
    return Z.var(0).sum() / Hc.var(0).sum()


print("保留的方差比例（真实数据）：")
print(f"  正确  V[:, -16:]              {var_frac(W_ok)*100:6.2f}%")
print(f"  正确  V[:, ::-1][:, :16]      {var_frac(W_rev)*100:6.2f}%")
print(f"  错误  V[:, ::-1][:, -16:]     {var_frac(W_bad)*100:6.2f}%   <- 取到了方差最小的 16 维")

M = np.abs(W_ok.T @ W_rev)
is_perm = np.allclose(M.sum(0), 1) and np.allclose(M.sum(1), 1) and np.isclose(M.max(), 1)
print(f"\n两种正确写法的关系：列顺序相反（正交置换矩阵），子空间完全相同 = {is_perm}")

rng = np.random.RandomState(0)
idx = rng.choice(n, 200, replace=False)
def dist(Z):
    d2 = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * Z @ Z.T
    return np.sqrt(np.maximum(d2, 0))


D1 = dist(Hc[idx] @ W_ok)
D2 = dist(Hc[idx] @ W_rev)
print(f"用两种写法算出的 200 个点两两距离最大差异：{np.abs(D1 - D2).max():.2e}（应为 0）")
D3 = dist(Hc[idx] @ W_bad)
print(f"错误写法与正确写法的距离平均差异：{np.abs(D1 - D3).mean():.2f}")
