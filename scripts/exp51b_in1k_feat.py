"""exp51b: 并行提取 ImageNet-1k 验证集（1000 类）的 DINOv2 特征

与 exp51 的差别：JPEG 用 draft 降采样解码（省掉大图逐像素重采样），多进程分摊。
前处理仍是 DINOv2 官方 ImageNet 协议：短边 256、中心裁 224、ImageNet 归一化。

输出：
  cache/feat_in1k.npy   50000 x 384 float32
  cache/in1k_y.npy      50000 int32
"""
import os
import glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data", "in1k_val")
SIZE = 224
SHORT = 256


def work(args):
    """单个 parquet 分片：解码 + 前向，返回 (Z, y)"""
    import io
    import torch
    import pandas as pd
    from PIL import Image

    f, dev = args
    torch.set_num_threads(2)

    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    phi = phi.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    df = pd.read_parquet(f)
    raws = [v["bytes"] for v in df["image"].values]
    y = df["label"].to_numpy().astype(np.int32)
    n = len(raws)
    Z = np.zeros((n, 384), np.float32)

    with torch.no_grad():
        for i in range(0, n, 96):
            chunk = raws[i:i + 96]
            arr = np.stack([prep(Image, io, b) for b in chunk])
            b_t = torch.from_numpy(arr).float().permute(0, 3, 1, 2).to(dev) / 255.0
            Z[i:i + 96] = phi((b_t - mean) / std).float().cpu().numpy()
    return Z, y


def prep(Image, io, raw):
    """draft 降采样解码 + 短边 256 + 中心裁 224"""
    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", (SHORT * 2, SHORT * 2))  # JPEG 直接按 1/2 或 1/4 解码
    im = im.convert("RGB")
    w, h = im.size
    if w < h:
        nw, nh = SHORT, int(round(h * SHORT / w))
    else:
        nh, nw = SHORT, int(round(w * SHORT / h))
    im = im.resize((nw, nh), Image.BILINEAR)
    left = (nw - SIZE) // 2
    top = (nh - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    return np.asarray(im, dtype=np.uint8)


def main():
    import torch
    from concurrent.futures import ProcessPoolExecutor

    out_f = os.path.join(CACHE, "feat_in1k.npy")
    out_y = os.path.join(CACHE, "in1k_y.npy")
    if os.path.exists(out_f) and os.path.exists(out_y):
        print(f"  已存在 {np.load(out_f).shape}", flush=True)
        return

    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    print(f"  parquet {len(files)} 个，cuda={torch.cuda.is_available()}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    parts = [(f, dev) for f in files]
    Zs, Ys = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, (Z, y) in enumerate(ex.map(work, parts)):
            Zs.append(Z)
            Ys.append(y)
            print(f"    完成 {i+1}/{len(files)}，累计 {sum(len(z) for z in Zs)} 张", flush=True)

    Z = np.concatenate(Zs, 0)
    y = np.concatenate(Ys, 0)
    np.save(out_f, Z)
    np.save(out_y, y)
    print(f"  完成 {Z.shape}，类数 {len(np.unique(y))}，"
          f"每类 {np.bincount(y).min()}~{np.bincount(y).max()}", flush=True)


if __name__ == "__main__":
    main()
