# MOLAR

Code for **MOLAR: Learning Multimodal Molecular Representations from Noisy Labels**.

## Data

The seven natural-noise benchmark datasets are included in `noise-7/`.

Each task folder contains:

```text
sd_train.csv
dr_val.csv
dr_test.csv
metadata.json
```

The controlled-noise raw datasets are included in `control-noise/raw/`:

```text
hiv
bace
bbbp
clintox
```

## Setup

Install the required Python packages in your own environment:

```bash
pip install torch torch-geometric rdkit pandas numpy scikit-learn pyyaml tqdm transformers
```

## Train One Task

Run from the project root:

```bash
python scripts/train_molar.py \
  --task MFPCBA-1053173-743445 \
  --config configs/config_molar.yaml \
  --data_root noise-7 \
  --device cuda \
  --seed 42
```

Use `--device cpu` if CUDA is unavailable.

## Train All Seven Tasks

```bash
python scripts/run_molar_gpu_parallel.py \
  --config configs/config_molar.yaml \
  --data_root noise-7 \
  --gpus 0 \
  --out_dir runs/molar_7tasks
```

For multiple GPUs:

```bash
python scripts/run_molar_gpu_parallel.py \
  --config configs/config_molar.yaml \
  --data_root noise-7 \
  --gpus 0,1 \
  --out_dir runs/molar_7tasks
```

## Evaluate A Checkpoint

```bash
python scripts/evaluate_molar_checkpoint.py \
  --config runs/molar_MFPCBA-1053173-743445_YYYYMMDD_HHMMSS/config.yaml \
  --checkpoint runs/molar_MFPCBA-1053173-743445_YYYYMMDD_HHMMSS/best_model.pt \
  --data_root noise-7 \
  --device cuda
```

## Controlled Noise

Prepare graph and text features for the four controlled-noise datasets:

```bash
python scripts/prepare_control_datasets.py \
  --root control-noise \
  --datasets hiv,bace,bbbp,clintox \
  --device cuda
```

For a lightweight CPU check without transformer embeddings:

```bash
python scripts/prepare_control_datasets.py \
  --root control-noise \
  --datasets hiv,bace,bbbp,clintox \
  --skip-embeddings
```

Generate fixed 5-fold 3:1:1 label-flip protocols:

```bash
python scripts/prepare_flip_labels.py \
  --data-root control-noise \
  --noise-ratio 0.30 \
  --num-folds 5 \
  --seed 42
```

Training outputs are saved under the selected `runs/` directory, including `best_model.pt`, `config.yaml`, and `results.json`.
