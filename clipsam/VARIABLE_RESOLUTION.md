# Variable Resolution SigLIP Support

## Overview

This feature enables the ClipSAM architecture to work with SigLIP encoders of different input resolutions, removing hardcoded assumptions about 384x384 input size and 27x27 spatial dimensions.

## Motivation

The original implementation was designed specifically for `google/siglip2-so400m-patch14-384` which:
- Takes 384×384 input images
- Uses 14×14 patches
- Produces 27×27 spatial feature maps (384÷14 ≈ 27)

However, SigLIP models come in various resolutions (224, 256, 384, 512, etc.), and the architecture had hardcoded spatial dimensions that prevented using different resolutions.

## Changes Made

### 1. Dynamic Resolution Detection

The model now automatically detects the SigLIP encoder's configuration:

```python
# Automatically extracted from the pretrained model
self.siglip_patch_size = self.siglip_config.patch_size
self.siglip_image_size = list(self.clip_vision_processor.size.values())[0]
self.siglip_spatial_dim = self.siglip_image_size // self.siglip_patch_size
```

**Example resolutions:**
- 224×224 with patch 16 → 14×14 spatial dim
- 256×256 with patch 16 → 16×16 spatial dim
- 384×384 with patch 14 → 27×27 spatial dim (original)
- 512×512 with patch 16 → 32×32 spatial dim

### 2. Dynamic Downsampling Calculation

The prompter network needs to downsample SigLIP features to a target spatial dimension (default 7×7 for SAM compatibility). The number of stride-2 downsampling steps is now calculated automatically:

```python
def _calculate_downsampling_steps(self, input_spatial_dim, target_spatial_dim):
    """
    Calculate required stride-2 downsampling steps.
    Each step reduces spatial dimension by half.
    """
    steps = 0
    current_dim = input_spatial_dim
    
    while current_dim > target_spatial_dim:
        current_dim = current_dim // 2
        steps += 1
    
    return steps
```

**Example calculations:**
- 27×27 → 7×7: needs 2 steps (27→13→6, close to 7)
- 32×32 → 7×7: needs 2 steps (32→16→8, close to 7)
- 14×14 → 7×7: needs 1 step (14→7)

### 3. Resolution-Agnostic Feature Processing

All hardcoded spatial dimensions in comments and reshape operations have been made dynamic:

**Before:**
```python
clip_visual_feat = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', 
                                   h=int(math.sqrt(clip_visual_feat.shape[1])))  # BX1152X27X27
```

**After:**
```python
clip_visual_feat = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', 
                                   h=self.siglip_spatial_dim, w=self.siglip_spatial_dim)
```

### 4. New Initialization Parameter

Added optional `target_spatial_dim` parameter to the model constructor:

```python
model = SigLipSamSegmentator(
    siglip_model_name='google/siglip-base-patch16-224',
    sam_model_name='facebook/sam-vit-base',
    target_spatial_dim=7  # Target output dimension for prompter network
)
```

## Usage

### Basic Usage (Automatic Configuration)

Simply specify a different SigLIP model and the architecture adapts automatically:

```python
from model import SigLipSamSegmentator

# 384x384 resolution (original)
model_384 = SigLipSamSegmentator(
    siglip_model_name='google/siglip2-so400m-patch14-384',
    sam_model_name='facebook/sam-vit-base'
)

# 224x224 resolution (if model exists)
model_224 = SigLipSamSegmentator(
    siglip_model_name='google/siglip-base-patch16-224',
    sam_model_name='facebook/sam-vit-base'
)
```

### Training with Different Resolutions

Update the training command to specify a different SigLIP model:

```bash
python train.py \
    --siglip_model google/siglip-base-patch16-224 \
    --sam_model facebook/sam-vit-base \
    --epochs 5 \
    --batch_size 4
```

### Testing Different Resolutions

Use the provided test script:

```bash
python test_variable_resolution.py
```

This will verify that the model correctly handles different resolutions.

## Architecture Details

### RSRefSeg Pipeline with Variable Resolution

```
Input Image (480×480)
    ↓
┌─────────────────────────────────────┐
│ SigLIP Vision Encoder               │
│ Resolution: {N}×{N} (configurable)  │
│ Patch Size: {P}                     │
│ Output: B×C×{N/P}×{N/P}            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Text-Visual Activation              │
│ - Local activation (token-level)    │
│ - Global activation (pooler-level)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Prompter Networks                   │
│ 1. Channel reduction (3C → 768)     │
│ 2. Positional embeddings            │
│ 3. Spatial downsampling             │
│    - Dynamic stride-2 convs         │
│    - Output: B×768×7×7             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ SAM Decoder                         │
│ Input: Downsampled features         │
│ Output: Segmentation mask           │
└─────────────────────────────────────┘
```

## Backward Compatibility

✅ **Fully backward compatible** - existing code continues to work without changes. The default behavior uses the original 384×384 model configuration.

## Known Limitations

1. **Model Availability**: Not all SigLIP resolution variants may be available on HuggingFace
2. **Downsampling Constraints**: Requires input spatial dimensions that can be downsampled to approximately 7×7
3. **Memory Usage**: Higher resolution models (512+) may require more GPU memory

## Testing

Run the test suite to verify functionality:

```bash
# Basic test with default 384×384 model
python test_variable_resolution.py

# Test with custom models (edit test_variable_resolution.py first)
# Uncomment test cases for different resolutions
python test_variable_resolution.py
```

## Future Enhancements

Potential improvements for future work:

1. **Adaptive Target Dimension**: Automatically adjust `target_spatial_dim` based on input resolution
2. **Multi-Scale Features**: Leverage multiple resolution scales for improved segmentation
3. **Resolution-Specific LoRA**: Train separate LoRA adapters for different resolutions
4. **Benchmark Suite**: Comprehensive performance comparison across resolutions

## References

- Original RSRefSeg paper: [Add citation if available]
- SigLIP: Sigmoid Loss for Language Image Pre-Training ([arXiv](https://arxiv.org/abs/2303.15343))
- Segment Anything (SAM): ([arXiv](https://arxiv.org/abs/2304.02643))
