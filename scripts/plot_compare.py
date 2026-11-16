# -*- coding: utf-8 -*-
"""对比三种降到 2 维的方式在坐标空间里是否保住了类别结构"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp2_sweep as E
import exp3_diag as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xtr, ytr, xte, yte = E.load_cifar()
node_classes, node_idx = E.make_shard2(ytr, 20, 10, 2, 0)
rng = np.random.RandomState(1)
tr_sel = []
for n, idx in enumerate(node_idx):
    for k in node_classes[n]:
        sel = idx[ytr[idx] == k]
        tr_sel.append(rng.choice(sel, size=min(300, len(sel)), replace=False))
tr_sel = np.concatenate(tr_sel)
H = np.load(os.path.join(E.CACHE, "feat_train.npy"))
y = ytr[tr_sel]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cmap = plt.get_cmap("tab10")
G = 8
for ax, (kind, title) in zip(axes, [("random", "random projection"),
                                    ("pca", "PCA"),
                                    ("lda", "LDA")]):
    a, _ = D.project(H, H[:1], kind, 2, y)
    z, _ = D.quantize_to_unit(a, a)
    for k in range(10):
        m = y == k
        ax.scatter(z[m, 0], z[m, 1], s=3, alpha=0.5, color=cmap(k))
    for g in range(1, G):
        ax.axvline(g / G, color="0.8", lw=0.5)
        ax.axhline(g / G, color="0.8", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  (grid {G}x{G})", fontsize=12)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
plt.tight_layout()
out = os.path.join(ROOT, "diag_projection_compare.png")
plt.savefig(out, dpi=150)
print("saved", out)
