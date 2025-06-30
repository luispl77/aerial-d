from transformers import pipeline
import torch
import argparse
import time
from PIL import Image

# Add argument parser for GPU selection
parser = argparse.ArgumentParser(description='Run Gemma 3 pipeline model')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

# Set device based on argument
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the local image using PIL
image_path = "./debug_output/P2532_patch_020952/P2532_patch_020952_obj_287764.png"
image = Image.open(image_path)
print(f"Loaded image from: {image_path}")

# Create pipeline with specified GPU
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device=device,
    use_fast=True,
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an advanced AI visual analysis assistant specializing in aerial and satellite imagery interpretation. Your expertise encompasses multiple domains including remote sensing, geographic information systems (GIS), urban planning, environmental monitoring, and geospatial analysis. You have been trained to recognize and analyze various features in aerial photographs including but not limited to: buildings and infrastructure (residential, commercial, industrial structures), transportation networks (roads, highways, railways, airports), natural features (forests, water bodies, agricultural fields, coastlines), urban development patterns, land use classifications, vegetation analysis, and environmental changes over time. When analyzing images, you should provide detailed, accurate descriptions that include spatial relationships, scale considerations, potential applications for urban planning or environmental monitoring, and any notable patterns or anomalies you observe. You should also consider the context of remote sensing applications, discussing how the observed features might be relevant for various analytical purposes such as change detection, urban growth monitoring, agricultural assessment, or disaster response planning. Please be thorough in your analysis while maintaining scientific accuracy and technical precision in your observations.\n\nTASK 2: Analyze the object's context and uniqueness factors:\n1. Examine the immediate surroundings of the object\n2. Identify distinctive features that could be used to uniquely identify this object:\n   - Nearby objects and their relationships\n   - Visual characteristics that distinguish it from similar objects\n   - Environmental context (roads, buildings, terrain) that provide reference points\n3. Consider how the original automated expressions could be improved\n4. Focus on features that would help someone locate this specific object without ambiguity"}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image},
            {"type": "text", "text": "Generate 5 unique referring expressions that could be used to identify the main object in this aerial image. Each expression should use different contextual cues, spatial relationships, and distinctive features to uniquely locate the object. Focus on creating expressions that would help someone find this specific object among similar objects in the scene."}
        ]
    }
]

# Time the generation
start_time = time.time()
output = pipe(text=messages, max_new_tokens=500)
end_time = time.time()

# Calculate metrics
generation_time = end_time - start_time
generated_content = output[0]["generated_text"][-1]["content"]

# Count tokens (approximation using whitespace split)
token_count = len(generated_content.split())

# Calculate tokens per second
tokens_per_second = token_count / generation_time if generation_time > 0 else 0

print(f"\n{'='*60}")
print(f"GENERATION METRICS:")
print(f"Time taken: {generation_time:.2f} seconds")
print(f"Approximate tokens generated: {token_count}")
print(f"Tokens per second: {tokens_per_second:.2f}")
print(f"{'='*60}\n")

print("GENERATED RESPONSE:")
print(generated_content) 