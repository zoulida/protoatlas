"""exp51c: 分片提取 ImageNet-1k 特征（单进程 + 增量保存）

起因：本机 GPU 前向异常慢（约 11 张/秒），全量 50000 张需 70 分钟以上。
改为逐分片处理、每片立即落盘，随时可停可续，避免一次失败全部重来。

用法：
  python exp51c_in1k_feat_inc.py --nfiles 6
"""
import os
import glob
import argparse
import numpy as np


def main():
    import io
    import time
    import torch
    import pandas as pd
    from PIL import Image

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE = os.path.join(ROOT, "cache")
    DATA = os.path.join(ROOT, "data", "in1k_val")
    SIZE, SHORT = 224, 256

    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=6)
    ap.add_argument("--bs", type=int, default=48)
    args = ap.parse_args()

    def prep(raw):
        im = Image.open(io.BytesIO(raw))
        im.draft("RGB", (SHORT * 2, SHORT * 2))
        im = im.convert("RGB")
        w, h = im.size
        if w < h:
            nw, nh = SHORT, int(round(h * SHORT / w))
        else:
            nh, nw = SHORT, int(round(w * SHORT / h))
        im = im.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
        return np.asarray(im.crop((left, top, left + SIZE, top + SIZE)), np.uint8)

    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))[:args.nfiles]
    print(f"处理 {len(files)} 个分片", flush=True)

    phi = torch.hub.load(os.path.join(ROOT, "third_party", "dinov2"),
                         "dinov2_vits14", source="local", pretrained=True)
    phi = phi.to("cuda").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    print("模型就绪", flush=True)

    for fi, f in enumerate(files):
        tag = os.path.basename(f)[:12]
        pf = os.path.join(CACHE, f"in1k_part_{fi:02d}.npz")
        if os.path.exists(pf):
            print(f"  [{fi+1}/{len(files)}] 已有，跳过", flush=True)
            continue
        t0 = time.time()
        df = pd.read_parquet(f)
        raws = [v["bytes"] for v in df["image"].values]
        y = df["label"].to_numpy().astype(np.int32)
        n = len(raws)
        Z = np.zeros((n, 384), np.float32)
        with torch.no_grad():
            for i in range(0, n, args.bs):
                arr = np.stack([prep(b) for b in raws[i:i + args.bs]])
                b_t = torch.from_numpy(arr).float().permute(0, 3, 1, 2).cuda() / 255.0
                Z[i:i + args.bs] = phi((b_t - mean) / std).float().cpu().numpy()
                if (i // args.bs) % 12 == 0:
                    print(f"    {i}/{n}  {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(pf, z=Z, y=y)
        print(f"  [{fi+1}/{len(files)}] {tag} 完成 {n} 张，用时 {time.time()-t0:.0f}s", flush=True)

    # 合并
    parts = sorted(glob.glob(os.path.join(CACHE, "in1k_part_*.npz")))
    Zs, Ys = [], []
    for p in parts:
        d = np.load(p)
        Zs.append(d["z"])
        Ys.append(d["y"])
    Z = np.concatenate(Zs, 0)
    y = np.concatenate(Ys, 0)
    np.save(os.path.join(CACHE, "feat_in1k.npy"), Z)
    np.save(os.path.join(CACHE, "in1k_y.npy"), y)
    cnt = np.bincount(y)
    print(f"合并完成 {Z.shape}，类数 {len(cnt)}，每类 {cnt.min()}~{cnt.max()}", flush=True)


if __name__ == "__main__":
    main()
