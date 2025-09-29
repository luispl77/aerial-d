# Using Fine-tuned SigLIP Checkpoint

## Your Fine-tuned Checkpoint

Location: `models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar`

This is a **SigLIP-Large (patch16-256)** model fine-tuned on remote sensing datasets:
- NWPU-RESISC45
- UCM (UC Merced)
- Sydney
- RSICD (Remote Sensing Image Captioning Dataset)
- RSITMD (Remote Sensing Image Text Matching Dataset)

## Training with Fine-tuned SigLIP

### Basic Command

```bash
python train.py \
    --siglip_model google/siglip2-large-patch16-256 \
    --siglip_checkpoint models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar \
    --epochs 5 \
    --batch_size 4 \
    --lr 1e-4
```

### Full Example with All Datasets

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --siglip_model google/siglip2-large-patch16-256 \
    --siglip_checkpoint models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar \
    --custom_name siglip_large_256_finetuned_rs \
    --epochs 5 \
    --batch_size 4 \
    --lr 1e-4 \
    --unique_only \
    --use_all_datasets
```

### With Historic Images

```bash
python train.py \
    --siglip_model google/siglip2-large-patch16-256 \
    --siglip_checkpoint models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar \
    --use_historic \
    --historic_percentage 20.0 \
    --epochs 5 \
    --batch_size 4
```

## Testing with Fine-tuned SigLIP

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name siglip_large_256_finetuned_rs \
    --siglip_model google/siglip2-large-patch16-256 \
    --siglip_checkpoint models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar \
    --dataset_type aeriald
```

## How It Works

1. **Base Model Loading**: 
   - Loads `google/siglip2-large-patch16-256` architecture from HuggingFace
   - Gets the processor and configuration

2. **Fine-tuned Weights Loading**:
   - Loads your checkpoint: `google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar`
   - Replaces the base weights with your fine-tuned weights
   - Uses `strict=False` to handle any key mismatches gracefully

3. **Variable Resolution Support**:
   - The model automatically detects: 256×256 resolution, patch size 16
   - Calculates spatial dimension: 256 ÷ 16 = 16×16
   - **Auto-adjusts target dimension**: 16×16 → 4×4 (instead of default 7×7)
   - Downsampling: 16×16 → 8×8 → 4×4 (2 stride-2 convolutions)

## Key Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--siglip_model` | HuggingFace model name for architecture | Yes |
| `--siglip_checkpoint` | Path to your `.pth.tar` checkpoint | Optional |
| `--custom_name` | Name for the training run | Recommended |
| `--epochs` | Number of training epochs | Yes |
| `--batch_size` | Batch size per GPU | Yes |

## Expected Output

When you run training, you should see:

```
Loading fine-tuned SigLIP weights from: models/google-siglip2-large-patch16-256-NWPU_UCM_Sydney_RSICD_RSITMD.pth.tar
✓ Successfully loaded fine-tuned SigLIP checkpoint
SigLIP Configuration:
  - Input resolution: 256x256
  - Patch size: 16
  - Output spatial dimension: 16x16
  - Target prompter output: 7x7
  - Auto-adjusting target_spatial_dim: 7 → 4 (input is 16x16)
  - Calculated downsampling steps: 2
```

**Note**: The model automatically adjusts the target dimension from 7×7 to 4×4 for the 256×256 resolution, since 16×16 cannot downsample to 7×7 cleanly.

## Notes

- The checkpoint is 3.3GB, so loading takes a few seconds
- Make sure you have enough GPU memory (SigLIP-Large is bigger than base)
- The fine-tuning on RS datasets should give better features for aerial/satellite images
- You can still use all the standard training options (LoRA, historic images, multi-dataset, etc.)

## Troubleshooting

If you get errors about missing keys or shape mismatches:
- The code uses `strict=False` to handle this automatically
- It will try to load as much as possible from the checkpoint
- Check the console output for warnings about which keys couldn't be loaded
