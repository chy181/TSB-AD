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

## Data Preparation

Download the univariate TSB-AD-U dataset from the official TSB-AD release:

```text
https://www.thedatum.org/datasets/TSB-AD-U.zip
```

Place and unzip it under the repository's `Datasets` directory:

```bash
mkdir -p Datasets
wget -O Datasets/TSB-AD-U.zip https://www.thedatum.org/datasets/TSB-AD-U.zip
unzip Datasets/TSB-AD-U.zip -d Datasets/
```


## Checkpoint Preparation

Download the MAFT checkpoints from Google Drive:

```text
https://drive.google.com/file/d/1cJCYxl5dGIm79tEDC2iGDaWU7_qUr3kv/view?usp=drive_link
```

Unzip the archive and place the files under:

```text
checkpoints/MAFT/
```

The directory should contain checkpoint files such as:

```text
checkpoints/MAFT/*_MultiAdapter_FT_win64_lr0.001.pt
checkpoints/MAFT/*_MultiAdapter_FT_win512_lr0.001.pt
```

Download the TimeRCD pretrained checkpoint from Hugging Face:

```bash
mkdir -p checkpoints/time-rcd
wget -O checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth \
  https://huggingface.co/thu-sail-lab/Time-RCD/resolve/main/best_model/pretrain_checkpoint_best_uni.pth
```

After preparation, the TimeRCD checkpoint should be located at:

```text
checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth
```

## Use Saved MAFT Checkpoints

Use checkpoint mode when you want to run MAFT from saved adapter weights:

```bash
cd /root/TSB-AD-Fork/TSB-AD
python benchmark_exp/HP_Tuning_U_MAFT.py \
  --limit 1 \
  --adapter_mode checkpoint \
  --device cuda:0 \
  --save_dir logs
```

For reference results, see:

```text
benchmark_exp/leaderboard_results/Uni_TimeRCD_MAFT.csv
```

## Run With On-The-Fly MAFT Training

Use train mode when you want to train the MAFT adapter from the prefix split encoded in the file name:

```bash
cd /root/TSB-AD-Fork/TSB-AD
python benchmark_exp/HP_Tuning_U_MAFT.py \
  --limit 1 \
  --adapter_mode train \
  --device cuda:0 \
  --save_dir logs
```
