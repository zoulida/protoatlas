"""测 DINOv2 ViT-S/14 在本机的提特征吞吐（用于估算千类增样本的代价）"""
import os
import io
import glob
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "in1k_val")
SIZE, SHORT = 224, 256


def prep(Image, raw):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if w < h:
        nw, nh = SHORT, int(round(h * SHORT / w))
    else:
        nh, nw = SHORT, int(round(w * SHORT / h))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    return np.asarray(im.crop((left, top, left + SIZE, top + SIZE)), dtype=np.uint8)


def main():
    import pandas as pd
    from PIL import Image

    t0 = time.time()
    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phi = phi.to(dev).eval()
    print(f"device={dev}  加载 {time.time()-t0:.1f}s  线程 {torch.get_num_threads()}", flush=True)

    f = sorted(glob.glob(os.path.join(DATA, "*.parquet")))[0]
    df = pd.read_parquet(f)
    raws = [v["bytes"] for v in df["image"].values][:256]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

    t1 = time.time()
    arr = np.stack([prep(Image, b) for b in raws])
    t_prep = time.time() - t1

    t2 = time.time()
    with torch.no_grad():
        for i in range(0, len(arr), 64):
            b = torch.from_numpy(arr[i:i + 64]).float().permute(0, 3, 1, 2).to(dev) / 255.0
            _ = phi((b - mean) / std)
    t_fw = time.time() - t2

    n = len(arr)
    print(f"{n} 张：预处理 {t_prep:.1f}s（{t_prep/n*1000:.0f} ms/张），"
          f"前向 {t_fw:.1f}s（{t_fw/n*1000:.0f} ms/张），合计 {t_prep/n*1000:.0f}+{t_fw/n*1000:.0f}", flush=True)
    per = (t_prep + t_fw) / n
    for tot, name in ((50000, "验证集 5 万张"), (100000, "10 万张"),
                      (250000, "25 万张"), (500000, "50 万张"), (1280000, "训练集 128 万张")):
        print(f"  {name}: {per*tot/3600:.1f} 小时", flush=True)


if __name__ == "__main__":
    main()
