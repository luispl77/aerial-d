import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os

# Completely disable torch compilation and dynamo
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False  # Disable cudnn compilation
torch.set_float32_matmul_precision('high')  # Suppress the warning

model = Gemma3ForConditionalGeneration.from_pretrained(
    #"google/gemma-3-4b-it",
    "gemma-aerial-referring",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"  # Changed from "sdpa" to "eager"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="left"
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
).to("cuda")

# Use torch.no_grad() to avoid any compilation during generation
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))