## ClipSAM: Training and Testing

Train and evaluate the SigLIP+SAM referring segmentation model on Aerial‑D and four additional datasets (RRSISD, RefSegRS, NWPU‑Refer, Urban1960SatBench).

### Setup
- Install dependencies from the repo root: `pip install -r ../requirements.txt`
- Dataset roots:
  - Aerial‑D: `../datagen/dataset`
  - RRSISD: `../datagen/rrsisd`
  - RefSegRS: `../datagen/refsegrs/RefSegRS`
  - NWPU‑Refer: `../datagen/NWPU-Refer`
  - Urban1960SatBench: `../datagen/Urban1960SatBench`

### Train on Aerial‑D
```bash
python train.py --epochs 5 --batch_size 4 --lr 1e-4

# Resume training (point to existing run folder)
python train.py --resume --custom_name <run_folder>
```

Expression selection (choose at most one):
```bash
python train.py --unique_only      # only the unique subset defined in the paper
python train.py --original_only    # only rule‑based original expressions
python train.py --enhanced_only    # only LLM‑enhanced expressions
```

### Train on all five datasets
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --custom_name all_datasets \
  --epochs 5 \
  --unique_only \
  --use_all_datasets
```

### Test on specific datasets
```bash
# AERIAL-D
CUDA_VISIBLE_DEVICES=0 python test.py --model_name all_datasets --dataset_type aeriald

# RRSISD
CUDA_VISIBLE_DEVICES=0 python test.py --model_name all_datasets --dataset_type rrsisd

# RefSegRS
CUDA_VISIBLE_DEVICES=0 python test.py --model_name all_datasets --dataset_type refsegrs

# NWPU-Refer
CUDA_VISIBLE_DEVICES=0 python test.py --model_name all_datasets --dataset_type nwpu

# Urban1960SatBench
CUDA_VISIBLE_DEVICES=0 python test.py --model_name all_datasets --dataset_type urban1960

# Visualizations only
python test.py --vis_only --num_vis 50
```

### Checkpoints and results
- Model folders under `./models/clip_sam_YYYYMMDD_HHMMSS_epochs{N}_bs{B}_lr{LR}`
- Results/visuals under `./results/<model_name>_<dataset>`

