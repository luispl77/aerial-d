#!/usr/bin/env python3
"""
Fine-tune Gemma 3 (4b or 12b) for aerial imagery referring expression generation using QLoRA
Combines aerial dataset from gemma3_sft_aerial.py with LoRA from gemma3_lora.py
"""

import torch
import argparse
import os
import json
import glob
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

# Prevent PEFT from trying to use bitsandbytes
import sys

# Mock bitsandbytes before importing peft to prevent the import error
class MockBNB:
    def __getattr__(self, name):
        return MockBNB()
    def __call__(self, *args, **kwargs):
        return MockBNB()

# Replace bitsandbytes module in sys.modules before importing peft
sys.modules['bitsandbytes'] = MockBNB()
sys.modules['bitsandbytes.nn'] = MockBNB()
sys.modules['bitsandbytes.nn.modules'] = MockBNB()

# Constants that match o3_enhance.py
NUM_ENHANCED = 1  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

def extract_model_size(model_id):
    """Extract model size from model ID (e.g., '4b' or '12b')"""
    if '4b' in model_id.lower():
        return '4b'
    elif '12b' in model_id.lower():
        return '12b'
    else:
        # Fallback: try to extract number followed by 'b'
        import re
        match = re.search(r'(\d+)b', model_id.lower())
        if match:
            return f"{match.group(1)}b"
        return 'unknown'

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma 3 model (4b or 12b) for aerial referring expressions with LoRA')
    parser.add_argument('--model_id', type=str, default="google/gemma-3-12b-it",
                       help='Hugging Face model ID')
    parser.add_argument('--enhanced_dir', type=str, default="./enhanced_annotations_o3",
                       help='Directory containing enhanced annotations')
    
    # Parse known args first to get model_id
    known_args, _ = parser.parse_known_args()
    model_size = extract_model_size(known_args.model_id)
    
    parser.add_argument('--output_dir', type=str, default=f"gemma-aerial-referring-{model_size}-lora",
                       help='Output directory for fine-tuned model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token')
    parser.add_argument('--merged_output_dir', type=str, default=f"gemma-aerial-{model_size}",
                       help='Output directory for merged full model')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (higher = more parameters, better adaptation)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha (scaling factor)')
    return parser.parse_args()

def setup_device_and_login(args):
    """Setup device and login to Hugging Face"""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check GPU capability for bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        print("Warning: GPU does not support bfloat16 optimally, but proceeding anyway.")
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        login(args.hf_token)
        print("Logged in to Hugging Face Hub")
    
    return device

def load_enhanced_dataset(enhanced_dir):
    """Load and prepare the enhanced aerial dataset"""
    print("Loading enhanced aerial dataset...")
    
    # System message that exactly matches o3_enhance.py
    system_message = (
        "You are an expert at creating natural language descriptions for objects and groups in aerial imagery. "
        "Your task is to help create diverse and precise referring expressions for the target highlighted with a red bounding box. "
        "The target may be a single object or a group/collection of multiple objects.\n\n"
        
        "IMPORTANT GUIDELINES:\n"
        "- If the original expressions refer to 'all', 'group of', or multiple objects, maintain this collective reference\n"
        "- If working with a group, use plural forms and consider the spatial distribution of the entire collection\n"
        "- If working with a single object, focus on that specific instance\n"
        "- Always preserve the scope and meaning of the original expressions\n"
        "- NEVER reference red boxes or markings in your expressions\n\n"
        
        "You have three tasks:\n\n"
        
        f"TASK 1: For each original expression listed below, create EXACTLY {NUM_ENHANCED} language variation that:\n"
        "1. MUST PRESERVE ALL SPATIAL INFORMATION from the original expression:\n"
        "   - Absolute positions (e.g., \"in the top right\", \"near the center\")\n"
        "   - Relative positions (e.g., \"to the right of\", \"below\")\n"
        "   - Collective scope (e.g., \"all\", \"group of\", individual references)\n"
        "2. Use natural, everyday language that a regular person would use\n"
        "   - Avoid overly formal or technical vocabulary\n"
        "   - Use common synonyms (e.g., \"car\" instead of \"automobile\")\n"
        "   - Keep the tone conversational and straightforward\n"
        "3. Ensure the variation uniquely identifies the target to avoid ambiguity\n"
        "4. Maintain the same scope as the original (single object vs. group/collection)\n\n"
        
        "TASK 2: Analyze the target's context and uniqueness factors:\n"
        "1. Examine the immediate surroundings of the target\n"
        "2. Identify distinctive features that could be used to uniquely identify the target:\n"
        "   - Nearby objects and their relationships\n"
        "   - Visual characteristics that distinguish it from similar objects\n"
        "   - Environmental context (roads, buildings, terrain) that provide reference points\n"
        "   - For groups: spatial distribution and arrangement patterns\n"
        "3. Consider how the original automated expressions could be improved\n"
        "4. Focus on features that would help someone locate this specific target without ambiguity\n\n"
        
        f"TASK 3: Generate EXACTLY {NUM_UNIQUE} new expressions that:\n"
        "1. MUST be based on one of the original expressions or their variations\n"
        "2. Add visual details ONLY when you are highly confident about them\n"
        "3. Each expression must uniquely identify the target\n"
        "4. Focus on describing the target's relationship with its immediate surroundings\n"
        "5. Maintain the core spatial information from the original expression\n"
        "6. Preserve the same scope as the original (individual vs. collective reference)\n\n"
        
        "You must return your output in the following JSON format:\n"
        "{\n"
        "  \"enhanced_expressions\": [\n"
        "    {\n"
        "      \"original_expression\": \"<original expression>\",\n"
        "      \"variation\": \"<single language variation>\"\n"
        "    },\n"
        "    ...\n"
        "  ],\n"
        "  \"unique_description\": \"<detailed analysis of spatial context and uniqueness factors>\",\n"
        "  \"unique_expressions\": [\n"
        "    \"<new expression based on original 1>\",\n"
        "    \"<new expression based on original 2>\"\n"
        "  ]\n"
        "}\n"
        "Only return the JSON object, no other text or comments.\n"
        "Write all the expressions using lowercase letters and no punctuation."
    )
    
    # Find all enhanced annotation files
    json_files = glob.glob(os.path.join(enhanced_dir, "*", "enhanced_expressions.json"))
    print(f"Found {len(json_files)} enhanced annotation files")
    
    formatted_dataset = []
    
    for json_file in json_files:
        try:
            # Load the JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Find corresponding image file
            obj_dir = os.path.dirname(json_file)
            image_files = glob.glob(os.path.join(obj_dir, "*.png"))
            if not image_files:
                print(f"Warning: No image found for {json_file}")
                continue
            
            image_path = image_files[0]
            
            # Create user prompt that includes the original expressions
            formatted_expressions = "\n".join([f"- {expr}" for expr in data["original_expressions"]])
            user_prompt = (
                f"Create language variations of the provided expressions while preserving spatial information, "
                f"analyze the spatial context for uniqueness factors, and generate new unique expressions for this {data['category']} "
                "(highlighted in red).\n\n"
                f"ORIGINAL EXPRESSIONS TO ENHANCE:\n{formatted_expressions}"
            )
            
            # Create the expected response JSON
            response_json = json.dumps(data["enhanced_data"], indent=2)
            
            # Convert to messages format
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": Image.open(image_path).convert("RGB"),
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response_json}],
                    },
                ],
            }
            
            formatted_dataset.append(sample)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(formatted_dataset)} samples")
    print("Sample data structure:")
    if formatted_dataset:
        print("System message preview:")
        print(formatted_dataset[0]["messages"][0]["content"][0]["text"][:200] + "...")
        print("\nUser prompt preview:")
        print(formatted_dataset[0]["messages"][1]["content"][1]["text"][:200] + "...")
    
    return formatted_dataset, system_message

def process_vision_info(messages):
    """Process vision information from messages"""
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

def setup_model_and_processor(args, device):
    """Setup model and processor with memory optimizations (no quantization)"""
    print(f"Loading model: {args.model_id}")
    
    # Define model init arguments with memory optimizations
    model_kwargs = dict(
        attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map=f"cuda:{args.gpu}",  # Explicit device mapping to avoid auto-detection issues
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True,  # Required for some models
    )

    # Load model and processor
    print("Loading model without quantization, using LoRA for memory efficiency...")
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    print("Model and processor loaded successfully")
    print(f"Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    return model, processor

def setup_peft_config(args):
    """Setup PEFT configuration for LoRA (without quantization)"""
    # Use specific target modules instead of "all-linear" to avoid triggering quantization detection
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj",     # MLP projections
    ]
    
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,  # Configurable alpha
        lora_dropout=0.1,  # Dropout for regularization
        r=args.lora_rank,  # Configurable rank
        bias="none",
        target_modules=target_modules,  # Specific modules instead of "all-linear"
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    print(f"LoRA config: rank={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
    print(f"Target modules: {target_modules}")
    trainable_params = peft_config.r * 2 * len(target_modules)  # More accurate estimate
    print(f"Estimated trainable parameters: ~{trainable_params}K")
    return peft_config

def create_collate_fn(processor):
    """Create collate function for training"""
    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens for not being used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch
    
    return collate_fn

def setup_training_config(args):
    """Setup training configuration with memory optimizations"""
    training_args = SFTConfig(
        output_dir=args.output_dir,                     # directory to save and repository id
        num_train_epochs=args.epochs,                   # number of training epochs
        per_device_train_batch_size=args.batch_size,    # batch size per device during training
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,                    # use gradient checkpointing to save memory
        optim="adamw_torch_fused",                      # use fused adamw optimizer
        logging_steps=1,                                # log every step for detailed monitoring
        save_strategy="no",                             # disable automatic checkpoint saving
        learning_rate=args.learning_rate,               # learning rate for LoRA
        bf16=True,                                      # use bfloat16 precision
        max_grad_norm=1.0,                              # gradient clipping
        warmup_ratio=0.03,                              # warmup ratio
        lr_scheduler_type="cosine",                     # cosine learning rate scheduler
        report_to="tensorboard",                        # report metrics to tensorboard
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # use reentrant checkpointing
        dataset_text_field="",                          # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        dataloader_pin_memory=False,                    # disable pin memory to save GPU memory
        dataloader_num_workers=0,                       # disable multiprocessing to save memory
        remove_unused_columns=False,                    # important for collator
    )
    
    print(f"Training config: batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    return training_args

def train_model(model, processor, dataset, peft_config, training_args):
    """Train the model using SFTTrainer"""
    print("Starting training...")
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    print("Training completed!")
    return trainer

def merge_and_save_model(args):
    """Merge adapter with base model and save full model"""
    print("Merging adapter with base model...")
    print("Note: This requires more than 30GB of CPU Memory")
    
    # Load base model
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, low_cpu_mem_usage=True)
    
    # Merge LoRA and base model and save
    peft_model = PeftModel.from_pretrained(model, args.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.merged_output_dir, safe_serialization=True, max_shard_size="2GB")
    
    processor = AutoProcessor.from_pretrained(args.output_dir)
    processor.save_pretrained(args.merged_output_dir)
    
    print(f"Full model merged and saved to '{args.merged_output_dir}' directory")



def main():
    args = parse_args()
    
    # Setup device and login
    device = setup_device_and_login(args)
    
    # Create dataset
    dataset, system_message = load_enhanced_dataset(args.enhanced_dir)
    
    if not dataset:
        print("No training data found!")
        return
    
    # Setup model and processor
    model, processor = setup_model_and_processor(args, device)
    
    # Setup PEFT config
    peft_config = setup_peft_config(args)
    
    # Setup training config
    training_args = setup_training_config(args)
    
    # Train model
    trainer = train_model(model, processor, dataset, peft_config, training_args)
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # Merge adapter with base model and save full model
    merge_and_save_model(args)
    
    print("Fine-tuning and model merging completed successfully!")

if __name__ == "__main__":
    main() 