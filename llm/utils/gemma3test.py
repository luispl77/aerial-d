import torch
import argparse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Add argument parser for GPU selection
parser = argparse.ArgumentParser(description='Run Gemma 3 model')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

# Set device based on argument
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="sdpa",
    cache_dir="./gemma_model",
    local_files_only=False
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="left",
    cache_dir="./gemma_model",
    local_files_only=False
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(device)

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))