# AerialD O3-500 Enhanced Dataset

## Overview

This dataset contains 500 OpenAI O3-enhanced referring expressions for aerial image segmentation, generated as part of the AerialSeg project. The expressions have been enhanced from rule-based annotations using OpenAI's O3 model to create more natural and diverse language descriptions for aerial imagery.

## Dataset Description

**Dataset Name**: `aeriald_o3_500`  
**Size**: 500 unique enhanced expressions  
**Source**: Rule-based annotations from AerialD dataset  
**Enhancement Method**: OpenAI O3 model via `o3_enhance_dual_image.py`  
**Format**: JSON annotations with referring expressions and dual image representations

## Usage in AerialSeg Project

### 1. AerialD Dataset Integration

This enhanced dataset serves as a high-quality subset of the larger AerialD dataset:

- **Training Data**: Used as premium training examples for referring segmentation models
- **Validation**: Provides natural language diversity for model evaluation
- **Benchmarking**: Serves as a quality reference for comparing rule-based vs. LLM-enhanced annotations

### 2. Gemma-Aerial-12B Fine-tuning

The dataset is specifically designed for fine-tuning the `gemma-aerial-12b` model:

**Fine-tuning Process**:
```bash
cd /cfs/home/u035679/aerialseg/llm
python gemma3_lora_finetune.py --enhanced_data_dir enhanced_annotations_o3_dual --model_name gemma-aerial-12b
```

**Training Configuration**:
- **LoRA Fine-tuning**: Parameter-efficient adaptation for aerial domain
- **Expression Enhancement**: Learns to generate natural language from rule-based descriptions
- **Domain Adaptation**: Specializes model for aerial imagery terminology and spatial relationships

### 3. Model Architecture Integration

The enhanced expressions are used with the ClipSAM architecture:

- **Input**: Natural language referring expressions
- **Processing**: SigLIP text encoder processes O3-enhanced descriptions
- **Output**: Precise segmentation masks for aerial objects

## File Structure

```
enhanced_annotations_o3_dual/
├── {PATCH_ID}_group_{GROUP_ID}/    # Group-based annotations
│   ├── {PATCH_ID}_clean.png        # Clean aerial image
│   ├── {PATCH_ID}_mask.png         # Segmentation mask overlay
│   └── enhanced_expressions.json   # O3-enhanced expressions
├── {PATCH_ID}_obj_{OBJ_ID}/        # Object-based annotations  
│   ├── {PATCH_ID}_bbox.png         # Bounding box visualization
│   ├── {PATCH_ID}_focused.png      # Object-focused view
│   └── enhanced_expressions.json   # O3-enhanced expressions
└── [500 total sample directories]
```

### Image Types Explained

- **Clean Images** (`*_clean.png`): Original aerial image patches without overlays
- **Mask Images** (`*_mask.png`): Images with colored segmentation mask overlays
- **Bbox Images** (`*_bbox.png`): Images with bounding box annotations
- **Focused Images** (`*_focused.png`): Cropped views focusing on the target object

## Key Features

- **Dual Image Representation**: Both clean images and visualization overlays for comprehensive understanding
- **Natural Language Diversity**: O3-enhanced expressions are more varied and natural than rule-based ones
- **Spatial Relationships**: Maintains precise spatial and relational information from original rules
- **Domain-Specific**: Tailored for aerial imagery with appropriate terminology
- **Quality Filtered**: Only unique, unambiguous expressions included
- **Multi-scale Annotations**: Supports both group-level and individual object-level referring expressions

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{aeriald_o3_500,
  title={AerialD O3-500: Enhanced Referring Expressions for Aerial Image Segmentation},
  author={Your Name},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aeriald_o3_500}
}
```

## Related Resources

- **Main Project**: [AerialSeg](https://github.com/luisml77/aerialseg)
- **Base Model**: [gemma-aerial-12b](https://huggingface.co/luisml77/gemma-aerial-12b)
- **Full AerialD Dataset**: Available in the main project repository

## License

This dataset is released under the same license as the AerialSeg project. Please refer to the main repository for licensing details.