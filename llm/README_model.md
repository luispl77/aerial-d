# Gemma-Aerial-12B: QLoRA Fine-tuned Model for Aerial Image Understanding

## Model Description

Gemma-Aerial-12B is a specialized language model fine-tuned for aerial image understanding and referring expression generation. Built on Google's Gemma-2-12B architecture, this model has been adapted using QLoRA (Quantized Low-Rank Adaptation) fine-tuning on the AerialD O3-500 enhanced dataset.

## Model Details

- **Base Model**: `google/gemma-2-12b-it`
- **Architecture**: Decoder-only transformer with 12B parameters
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Training Data**: [AerialD O3-500](https://huggingface.co/datasets/luisml77/aeriald_o3_500) - 500 O3-enhanced referring expressions
- **Domain**: Aerial imagery, satellite images, remote sensing
- **Task**: Referring expression generation and understanding for aerial segmentation

## Training Details

### QLoRA Configuration
- **LoRA Rank (r)**: 64
- **LoRA Alpha**: 16
- **LoRA Dropout**: 0.1
- **Target Modules**: Query, Key, Value, and Output projection layers
- **Quantization**: 4-bit NF4 with double quantization
- **Compute dtype**: bfloat16

### Training Parameters
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (effective batch size with gradient accumulation)
- **Training Steps**: Optimized for convergence on 500 samples
- **Optimizer**: AdamW with cosine learning rate schedule
- **Gradient Clipping**: 1.0

### Dataset Enhancement Pipeline
The model was trained on expressions enhanced through:
1. **Rule-based Generation**: Spatial relationships and object properties
2. **OpenAI O3 Enhancement**: Natural language diversification via `o3_enhance_dual_image.py`
3. **Quality Filtering**: Unique, unambiguous expressions only

## Capabilities

### Aerial Domain Expertise
- **Spatial Reasoning**: Understanding of aerial perspective and spatial relationships
- **Object Recognition**: Familiarity with aerial imagery objects (buildings, roads, vegetation, etc.)
- **Geographic Terminology**: Appropriate use of remote sensing and geographic terms
- **Scale Awareness**: Understanding of different zoom levels and resolutions

### Referring Expression Tasks
- **Expression Generation**: Create natural language descriptions for aerial objects
- **Spatial Relationships**: Express relative positions using aerial-specific language
- **Multi-scale Descriptions**: Handle both individual objects and grouped regions
- **Contextual Understanding**: Incorporate surrounding context in descriptions

## Usage

### Basic Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("luisml77/gemma-aerial-12b")
model = AutoModelForCausalLM.from_pretrained(
    "luisml77/gemma-aerial-12b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate referring expression for aerial image
prompt = "Describe the highlighted building in this aerial image:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Integration with AerialSeg Pipeline
```python
# Use with ClipSAM for referring segmentation
from aerialseg.clipsam import SigLipSamSegmentator

segmentator = SigLipSamSegmentator()
enhanced_expression = model.generate_expression(rule_based_input)
segmentation_mask = segmentator.segment(image, enhanced_expression)
```

## Performance

### Training Metrics
- **Final Training Loss**: Optimized for aerial domain expressions
- **Validation Perplexity**: Improved on aerial-specific vocabulary
- **Expression Quality**: Enhanced naturalness compared to rule-based baseline

### Evaluation
- **Domain Adaptation**: Successfully adapted to aerial imagery terminology
- **Expression Diversity**: Generates varied, natural language descriptions
- **Spatial Accuracy**: Maintains precise spatial relationship information

## Fine-tuning Script

The model was fine-tuned using the provided script:
```bash
cd /cfs/home/u035679/aerialseg/llm
python gemma3_lora_finetune.py \
    --enhanced_data_dir enhanced_annotations_o3_dual \
    --model_name gemma-aerial-12b \
    --output_dir ./gemma-aerial-12b \
    --lora_r 64 \
    --lora_alpha 16
```

## Model Architecture

```
Gemma-2-12B Base Model
├── Embedding Layer (Frozen)
├── 42 Transformer Blocks
│   ├── Multi-Head Attention (LoRA Adapted)
│   │   ├── Q, K, V Projections (LoRA)
│   │   └── Output Projection (LoRA)
│   ├── Feed-Forward Network (Frozen)
│   └── Layer Normalization (Frozen)
├── Output Head (Frozen)
└── QLoRA Parameters: ~167M trainable parameters
```

## Limitations

- **Training Data Size**: Limited to 500 enhanced samples
- **Domain Specificity**: Optimized for aerial imagery, may not generalize to other domains
- **Language Support**: Primarily English expressions
- **Resolution Dependency**: Performance may vary with different image resolutions

## Ethical Considerations

- **Dual-Use Applications**: Model outputs should be used responsibly for civilian applications
- **Privacy**: Be mindful of privacy when processing aerial imagery of populated areas
- **Accuracy**: Model outputs should be validated for critical applications

## Citation

```bibtex
@model{gemma_aerial_12b,
  title={Gemma-Aerial-12B: QLoRA Fine-tuned Model for Aerial Image Understanding},
  author={Your Name},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/luisml77/gemma-aerial-12b}
}
```

## Related Resources

- **Training Dataset**: [AerialD O3-500](https://huggingface.co/datasets/luisml77/aeriald_o3_500)
- **Base Model**: [Gemma-2-12B-IT](https://huggingface.co/google/gemma-2-12b-it)
- **Project Repository**: [AerialSeg](https://github.com/luisml77/aerialseg)
- **Research Paper**: Available in the thesis documentation

## License

This model follows the same licensing terms as the base Gemma-2-12B model. Please refer to Google's Gemma license for usage terms and conditions.

## Acknowledgments

- **Google**: For the base Gemma-2-12B model architecture
- **OpenAI**: For O3 enhancement of training expressions
- **Hugging Face**: For the transformers library and model hosting
- **AerialSeg Project**: For the comprehensive aerial segmentation framework