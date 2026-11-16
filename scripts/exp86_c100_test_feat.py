# -*- coding: utf-8 -*-
"""提取 CIFAR-100 测试集全部 10 000 张的 DINOv2 特征。

原来只缓存了 3 000 张（cache/feat_c100.npy），中间类数档（K=20、50）
需要按类均衡抽样，样本不够。本脚本产出 cache/feat_c100_test10k.npy
与 cache/c100_test10k.npy（标签）。
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def main():
    import pandas as pd
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from exp1_route import CFG

    raw = os.path.join(CACHE, "c100_test10k.npz")
    if os.path.exists(raw):
        d = np.load(raw)
        X, y = d["x"], d["y"]
        log(f"原始缓存 {X.shape}")
    else:
        df = pd.read_parquet(os.path.join(DATA, "cifar100_test.parquet"))
        n = len(df)
        X = np.zeros((n, 32, 32, 3), np.uint8)
        col = "img" if "img" in df.columns else "image"
        for i in range(n):
            im = Image.open(io.BytesIO(df[col].values[i]["bytes"])).convert("RGB")
            if im.size != (32, 32):
                im = im.resize((32, 32), Image.BICUBIC)
            X[i] = np.array(im)
            if (i + 1) % 5000 == 0:
                log(f"  解码 {i+1}/{n}")
        y = df["fine_label"].to_numpy().astype(np.int32)
        np.savez_compressed(raw, x=X, y=y)
        log(f"原始缓存完成 {X.shape}")

    out = os.path.join(CACHE, "feat_c100_test10k.npy")
    if os.path.exists(out):
        log(f"特征已存在 {np.load(out).shape}")
        return

    torch.set_num_threads(8)
    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    phi.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    Z = np.zeros((len(X), 384), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 64):
            b = torch.from_numpy(X[i:i + 64]).float().permute(0, 3, 1, 2) / 255.0
            b = F.interpolate(b, size=CFG["img_size"], mode="bicubic",
                              align_corners=False)
            Z[i:i + 64] = phi((b - mean) / std).numpy()
            if (i // 64) % 20 == 0:
                log(f"  特征 {i}/{len(X)}")
    np.save(out, Z)
    log(f"完成 {Z.shape}")


if __name__ == "__main__":
    main()
