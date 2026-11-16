"""提取 CIFAR-100 训练集 DINOv2 特征（5 万张），供 100 类扩展实验用。

缓存到 cache/feat_c100_train.npy 与 cache/c100_train.npz（标签 + 超类标签）。
"""
import os
import io
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)


def log(*a):
    print(*a, flush=True)


def main():
    import pandas as pd
    from PIL import Image

    raw_cache = os.path.join(CACHE, "c100_train.npz")
    if os.path.exists(raw_cache):
        d = np.load(raw_cache)
        X, y, coarse = d["x"], d["y"], d["coarse"]
    else:
        df = pd.read_parquet(os.path.join(DATA, "cifar100_train.parquet"))
        log(f"  parquet 载入 {len(df)} 行，列 {df.columns.tolist()}")
        n = len(df)
        X = np.zeros((n, 32, 32, 3), np.uint8)
        col = "img" if "img" in df.columns else "image"
        for i in range(n):
            im = Image.open(io.BytesIO(df[col].values[i]["bytes"])).convert("RGB")
            if im.size != (32, 32):
                im = im.resize((32, 32), Image.BICUBIC)
            X[i] = np.array(im)
            if (i + 1) % 10000 == 0:
                log(f"    解码 {i+1}/{n}")
        y = df["fine_label"].to_numpy().astype(np.int32)
        coarse = df["coarse_label"].to_numpy().astype(np.int32)
        np.savez_compressed(raw_cache, x=X, y=y, coarse=coarse)
        log(f"  原始缓存完成 {X.shape}")

    out = os.path.join(CACHE, "feat_c100_train.npy")
    if os.path.exists(out):
        log(f"  特征已存在 {np.load(out).shape}")
        return

    import torch.nn.functional as F
    from exp1_route import CFG

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
            b = F.interpolate(b, size=CFG["img_size"], mode="bicubic", align_corners=False)
            Z[i:i + 64] = phi((b - mean) / std).numpy()
            if (i // 64) % 100 == 0:
                log(f"    特征 {i}/{len(X)}")
    np.save(out, Z)
    log(f"  完成 {Z.shape}")


if __name__ == "__main__":
    main()
