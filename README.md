# ProtoAtlas

Reference implementation and experiment code for the manuscript

> **ProtoAtlas: Feature Addressing and Collaborative Classification in Class-Heterogeneous Open Networks**

ProtoAtlas separates *deciding* from *addressing*. Each node keeps a complete
classifier locally and registers one coordinate per class it can serve; the
coordinate is the mean position of that class's local samples under a shared,
frozen encoder. All (node, class) pairs, called *vnodes*, are linked into an
overlay by coordinate proximity. A query is first self-assessed locally; if the
local model cannot decide, only the coordinate travels across the overlay, hop
by hop, until the nearest vnode is reached and the node that registered it
answers. No parameter is averaged, no global label space is required, and the
routing side carries nothing that needs training.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | All experiment scripts, named `expNN_*.py`; higher numbers are newer |
| `results/` | JSON output of the scripts, one file per experiment |
| `figs/` | Plotting scripts that regenerate every figure from `results/` |
| `third_party/dinov2/` | DINOv2 source, loaded locally by `torch.hub` (Apache-2.0) |
| `data/` | **not included** — raw datasets, see below |
| `cache/` | **not included** — extracted features, see below |

## Environment

Python 3.13 was used for all reported runs.

```bash
pip install -r requirements.txt
```

`torch 2.6.0+cu124` was used. Install the wheel matching your CUDA driver; a
CPU-only build runs as well but makes feature extraction slow.

## Data and feature cache

Neither the raw datasets nor the extracted features are stored in this
repository. Place them as follows.

```
data/
  cifar100_train.parquet     # CIFAR-100 train, columns include fine_label
  cifar100_test.parquet      # CIFAR-100 test,  columns include fine_label
cache/
  feat_c100_train.npy        # 50000 x 384, DINOv2 features of CIFAR-100 train
  c100_train.npz             # labels ("y") for the above
  feat_c100_test10k.npy      # 10000 x 384, CIFAR-100 test features
  c100_test10k.npz           # labels for the above
  feat_in1k.npy              # 50000 x 384, ImageNet-1k val features
  in1k_y.npy                 # labels for the above
```

**Option A (recommended).** Download the prepared feature cache from the Zenodo
record linked in the manuscript's data availability statement, and unzip it into
`cache/`. This skips GPU feature extraction entirely.

**Option B.** Regenerate the cache from the raw data. CIFAR-100 can be obtained
from <https://www.cs.toronto.edu/~kriz/cifar.html> and converted to parquet;
ImageNet-1k validation features were extracted from the `mrm8488/ImageNet1K-val`
split (50000 images, 50 per class), using the official DINOv2 preprocessing
(short side 256, centre crop 224, ImageNet normalisation).

```bash
python scripts/c100_prep.py            # -> feat_c100_train.npy, c100_train.npz
python scripts/exp86_c100_test_feat.py # -> feat_c100_test10k.npy, c100_test10k.npz
python scripts/exp51_in1k_feat.py      # -> feat_in1k.npy, in1k_y.npy
```

## Reproducing the reported results

Every script resolves `ROOT` as its own parent directory, so `results/`,
`cache/` and `third_party/` are located relative to the repository, independent
of the working directory.

```bash
python scripts/exp85_end2end_Kscan.py     # end-to-end, class-count sweep
python scripts/exp28b_greedy_lda.py       # multi-start greedy routing
python scripts/exp97_dedup_kscan_c100.py  # per-class de-duplication
python scripts/exp52_scale_route.py       # routing at 10^3 classes, 10^4 nodes
python scripts/exp95_selfcheck_energy.py  # local self-assessment readout
```

Several scripts take environment variables that control their scale, for
example `KSCAN=20 SEEDS=1,2,3 python scripts/exp85_end2end_Kscan.py` or
`NTEST=10000 python scripts/exp95_selfcheck_energy.py`.

| Manuscript item | Script | Result file |
|---|---|---|
| Table 5-1 / Fig. 5-1 (end-to-end, four routes) | `exp85_end2end_Kscan.py` | `results_end2end_Kscan.json` |
| Fig. 5-2 (multi-start and diffusion) | `exp28b_greedy_lda.py` | `results_greedy_grid_lda.json` |
| Fig. 5-3 (per-class de-duplication) | `exp97_dedup_kscan_c100.py` | `results_dedup_kscan_c100.json` |
| Fig. 5-4 / App. H (scale) | `exp52_scale_route.py` | `results_scale_route.json` |
| App. G (self-assessment) | `exp95_selfcheck_energy.py` | `results_selfcheck_energy_10000.json` |
| App. A (discriminative projection) | `exp78_*.py` | `results_pca_*.json` |

Figures are regenerated from the JSON files:

```bash
python figs/fig5_1_kscan.py   # and the other scripts in figs/
```

## License and third-party code

This repository is released under the Apache License 2.0 (see `LICENSE`).

`third_party/dinov2/` is a copy of the DINOv2 reference implementation by Meta
Platforms, Inc., redistributed under the Apache License 2.0. Its own `LICENSE`
and model licences are kept in place; see `third_party/dinov2/`.

The datasets used here — CIFAR-10, CIFAR-100 and ImageNet-1k — are not
redistributed and remain under their original terms.
