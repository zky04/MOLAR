# MOLAR

Code for **MOLAR: Learning Multimodal Molecular Representations from Noisy Labels**.

## Train All Tasks

Natural-noise tasks:

```bash
python scripts/run_molar_gpu_parallel.py \
  --config configs/config_molar.yaml \
  --data_root noise-7 \
  --gpus 0,1 \
  --out_dir runs/molar_natural
```

Controlled-noise tasks:

```bash
python scripts/prepare_control_datasets.py \
  --root control-noise \
  --datasets hiv,bace,bbbp,clintox \
  --device cuda

python scripts/prepare_flip_labels.py \
  --data-root control-noise \
  --datasets hiv,bace,bbbp,clintox \
  --noise-ratio 0.30 \
  --num-folds 5 \
  --seed 42
```
