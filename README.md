# MAFT: Multi-Adapter Enhanced Fine-Tuning of Time Series Foundation Models for Anomaly Detection

This document describes how to use MAFT in [TSB-AD-U](https://thedatumorg.github.io/TSB-AD/).



## Code Structure

Important files and directories:

```text
/root/TSB-AD-Fork/TSB-AD/
├── benchmark_exp/
│   ├── HP_Tuning_U_MA.py
│   └── leaderboard_results/
│       └── Uni_TimeRCD_MAFT.csv
├── checkpoints/
│   ├── MAFT/
│   │   ├── *_MultiAdapter_FT_win64_lr0.001.pt
│   │   └── *_MultiAdapter_FT_win512_lr0.001.pt
│   └── time-rcd/
│       └── pretrain_checkpoint_best_uni.pth
├── Datasets/
│   ├── File_List/
│   │   ├── TSB-AD-U-Eva.csv
│   │   └── TSB-AD-U-Eva-Full.csv
│   └── TSB-AD-U/
├── TSB_AD/
│   └── models/
│       ├── TimeRCD_MAFT.cpython-311-x86_64-linux-gnu.so
│       └── Time_RCD.py
├── requirements.txt
└── setup.py
```

The MAFT implementation used by the benchmark is packaged as:

```text
TSB_AD/models/TimeRCD_MAFT.cpython-311-x86_64-linux-gnu.so
```

The benchmark script imports it with:

```python
from TSB_AD.models.TimeRCD_MAFT import TimeRCD_MAFT
```

## Environment

Use Python 3.11. The provided `.so` file was compiled for CPython 3.11 on Linux x86_64, so the Python version must match.

Recommended setup:

```bash
cd /root/TSB-AD-Fork/TSB-AD
conda create -n TSB-AD python=3.11
conda activate TSB-AD
pip install -r requirements.txt
pip install -e .
```

If PyTorch is not installed correctly for CUDA, install a CUDA-compatible PyTorch build first, then rerun the package install.

Quick import check:

```bash
python -c "import TSB_AD.models.TimeRCD_MAFT as m; print(m.__file__)"
```

Expected output should end with:

```text
TSB_AD/models/TimeRCD_MAFT.cpython-311-x86_64-linux-gnu.so
```

## Usage Script

The main entry point is:

```text
benchmark_exp/HP_Tuning_U_MA.py
```

It runs TimeRCD + MAFT score fusion on the univariate TSB-AD evaluation list.

The script reads file names from:

```text
Datasets/File_List/TSB-AD-U-Eva.csv
```

The default dataset directory is:

```text
Datasets/TSB-AD-U
```

## Use Saved MAFT Checkpoints

Use checkpoint mode when you want to run MAFT from saved adapter weights:

```bash
cd /root/TSB-AD-Fork/TSB-AD
python benchmark_exp/HP_Tuning_U_MA.py \
  --limit 1 \
  --adapter_mode checkpoint \
  --device cuda:0 \
  --save_dir logs
```

## Run With On-The-Fly MAFT Training

Use train mode when you want to train the MAFT adapter from the prefix split encoded in the file name:

```bash
cd /root/TSB-AD-Fork/TSB-AD
python benchmark_exp/HP_Tuning_U_MA.py \
  --limit 1 \
  --adapter_mode train \
  --device cuda:0 \
  --save_dir logs
```


## Key Arguments

```text
--limit
    Number of datasets to run. Use 1 for a smoke test. Use 0 or omit it for all files.

--adapter_mode
    train: train MAFT from the prefix split.
    checkpoint: load MAFT weights from checkpoints/MAFT.
    score: load cached MAFT scores if available.
    auto: checkpoint first, then score fallback.

--device
    CUDA device string, for example cuda:0 or cuda:6.

--save_dir
    Directory for CSV and summary outputs.

--adapter_checkpoint_dir
    MAFT checkpoint directory. Default: checkpoints/MAFT.

--timercd_checkpoint
    TimeRCD checkpoint. Default: checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth.
```

## Troubleshooting

If import fails, confirm the `.so` is loaded:

```bash
python -c "import TSB_AD.models.TimeRCD_MAFT as m; print(m.__file__)"
```

If checkpoint mode fails, check that the corresponding file exists under:

```text
checkpoints/MAFT/
```

If CUDA device selection looks unexpected, check available GPUs:

```bash
nvidia-smi
```

If the Python version is not 3.11, rebuild the `.so` for the active Python environment or switch to the matching `TSB-AD` conda environment.
