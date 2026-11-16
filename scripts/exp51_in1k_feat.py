"""exp51: 提取 ImageNet-1k 验证集（1000 类）的 DINOv2 特征

用途：为 K=10^3 档提供与 CIFAR-100 同口径的特征。
前处理按 DINOv2 官方 ImageNet 协议：短边缩到 256，中心裁 224，ImageNet 归一化。
数据源：mrm8488/ImageNet1K-val（50000 张，每类 50 张，标签 0..999）。

输出：
  cache/feat_in1k.npy   50000 x 384 float32
  cache/in1k_y.npy      50000 int32
"""
import os
import io
import glob
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data", "in1k_val")
SIZE = 224
SHORT = 256


def log(*a):
    print(*a, flush=True)


def prep_image(Image, raw):
    """短边缩到 256，中心裁 224，返回 HWC uint8"""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if w < h:
        nw, nh = SHORT, int(round(h * SHORT / w))
    else:
        nh, nw = SHORT, int(round(w * SHORT / h))
    im = im.resize((nw, nh), Image.BICUBIC)
    left = (nw - SIZE) // 2
    top = (nh - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    return np.asarray(im, dtype=np.uint8)


def main():
    import pandas as pd
    from PIL import Image

    out_f = os.path.join(CACHE, "feat_in1k.npy")
    out_y = os.path.join(CACHE, "in1k_y.npy")
    if os.path.exists(out_f) and os.path.exists(out_y):
        log(f"  已存在 {np.load(out_f).shape}")
        return

    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    log(f"  parquet {len(files)} 个")

    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi = phi.to(dev).eval()
    log(f"  device={dev}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    Zs, Ys = [], []
    done = 0
    with torch.no_grad():
        for fi, f in enumerate(files):
            df = pd.read_parquet(f)
            raws = [v["bytes"] for v in df["image"].values]
            y = df["label"].to_numpy().astype(np.int32)
            n = len(raws)
            Z = np.zeros((n, 384), np.float32)
            for i in range(0, n, 64):
                chunk = raws[i:i + 64]
                arr = np.stack([prep_image(Image, b) for b in chunk])
                b_t = torch.from_numpy(arr).float().permute(0, 3, 1, 2).to(dev) / 255.0
                Z[i:i + 64] = phi((b_t - mean) / std).float().cpu().numpy()
            Zs.append(Z)
            Ys.append(y)
            done += n
            log(f"    {fi+1}/{len(files)} 文件，累计 {done} 张")

    Z = np.concatenate(Zs, 0)
    y = np.concatenate(Ys, 0)
    np.save(out_f, Z)
    np.save(out_y, y)
    log(f"  完成 {Z.shape}，类数 {len(np.unique(y))}，每类 {np.bincount(y).min()}~{np.bincount(y).max()}")


if __name__ == "__main__":
    main()
