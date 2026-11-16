"""exp67a: 取 ImageNet-1k 前 100 类，训练集做数据增广并提特征

目的：千类档每类只有 50 张（验证集），样本量把 body 压死。本脚本用增广把
每类训练样本放大到 25×N 张，用来检验「千类尺度下增加样本，body 是否开始有用」。

划分（每类 50 张）：
  前 25 张 → 训练，各做 N 种增广（第 0 种为原图的标准中心裁）
  后 25 张 → 测试，用 cache/feat_in1k.npy 里现成的特征（未增广，无泄漏）

增广：随机 resized-crop（面积比 0.4–1.0、长宽比 3/4–4/3）后缩到 224，
      水平翻转 p=0.5，亮度/对比度抖动 ±0.2。

输出 cache/feat_in1k100_aug{N}.npy（形状 100*25*N × 384）与同目录标签
cache/in1k100_aug{N}_y.npy。
"""
import os
import io
import sys
import time
import glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data", "in1k_val")
SIZE = 224
NCLS = 100
N_TR = 25


def log(*a):
    print(*a, flush=True)


def crop_aug(Image, raw, rng, aug_id):
    """aug_id=0 走标准中心裁；其余走随机 resized-crop + 翻转 + 亮度对比度"""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if aug_id == 0:
        short = 256
        if w < h:
            nw, nh = short, int(round(h * short / w))
        else:
            nh, nw = short, int(round(w * short / h))
        im = im.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
        return np.asarray(im.crop((left, top, left + SIZE, top + SIZE)), np.uint8)

    from PIL import ImageEnhance
    area = w * h
    for _ in range(10):
        scale = rng.uniform(0.4, 1.0)
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw = int(round(np.sqrt(area * scale * ratio)))
        ch = int(round(np.sqrt(area * scale / ratio)))
        if cw <= w and ch <= h:
            break
    else:
        cw, ch = min(w, h), min(w, h)
    left = rng.randint(0, w - cw + 1)
    top = rng.randint(0, h - ch + 1)
    im = im.crop((left, top, left + cw, top + ch))
    if rng.rand() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    if rng.rand() < 0.7:
        im = ImageEnhance.Brightness(im).enhance(1.0 + rng.uniform(-0.2, 0.2))
        im = ImageEnhance.Contrast(im).enhance(1.0 + rng.uniform(-0.2, 0.2))
    return np.asarray(im, np.uint8)


def main():
    import pandas as pd
    from PIL import Image
    import torch

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    c0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    c1 = int(sys.argv[3]) if len(sys.argv) > 3 else NCLS
    part = (c0, c1) != (0, NCLS)
    out_f = os.path.join(CACHE, f"feat_in1k100_aug{N}.npy" if not part
                         else f"feat_in1k100_aug{N}_p{c0}_{c1}.npy")
    out_y = os.path.join(CACHE, f"in1k100_aug{N}_y.npy" if not part
                         else f"in1k100_aug{N}_y_p{c0}_{c1}.npy")
    if os.path.exists(out_f) and os.path.exists(out_y):
        log(f"已存在 {np.load(out_f).shape}")
        return

    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    raws_by_cls = {c: [] for c in range(NCLS)}
    for f in files:
        df = pd.read_parquet(f)
        rr = [v["bytes"] for v in df["image"].values]
        yy = df["label"].to_numpy().astype(np.int32)
        for r, yv in zip(rr, yy):
            if yv < NCLS:
                raws_by_cls[int(yv)].append(r)
        if all(len(v) >= 50 for v in raws_by_cls.values()):
            break
    log(f"每类已收集 {min(len(v) for v in raws_by_cls.values())} 张")

    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi = phi.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    log(f"device={dev}，每类 {N_TR} 张 × {N} 增广 × {NCLS} 类 = {N_TR*N*NCLS} 张")

    Zs, Ys = [], []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for c in range(c0, c1):
            for aug in range(N):
                rng = np.random.RandomState(1000 * c + aug)
                arr = np.stack([crop_aug(Image, r, rng, aug) for r in raws_by_cls[c][:N_TR]])
                b = torch.from_numpy(arr).float().permute(0, 3, 1, 2).to(dev) / 255.0
                Zs.append(phi((b - mean) / std).float().cpu().numpy())
                Ys.append(np.full(len(arr), c, np.int32))
            done += N_TR * N
            if (c + 1 - c0) % 5 == 0:
                el = time.time() - t0
                log(f"  {c+1}/{c1} 类，{done} 张，{el/60:.1f} 分钟，"
                    f"预计还需 {el/(c+1-c0)*(c1-c-1)/60:.1f} 分钟")

    Z = np.concatenate(Zs, 0).astype(np.float32)
    y = np.concatenate(Ys, 0)
    np.save(out_f, Z)
    np.save(out_y, y)
    log(f"写入 {out_f} {Z.shape}，用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
