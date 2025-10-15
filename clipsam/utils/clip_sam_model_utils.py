import os
import torch
import sys

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def load_model(model_type, checkpoint_path, gpu_id=0, sam_model_name='facebook/sam-vit-base'):
    """Load the CLIP-SAM model from checkpoint."""
    if model_type != 'clip_sam':
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Import here to avoid circular imports
    from model import SigLipSamSegmentator
    
    # Initialize model with default parameters
    model = SigLipSamSegmentator(
        siglip_model_name='google/siglip2-so400m-patch14-384',
        sam_model_name=sam_model_name,
        down_spatial_times=2,
        with_dense_feat=True,
        device='cpu'  # First load to CPU
    )
    
    # Load checkpoint to CPU first
    device = torch.device(f'cuda:{gpu_id}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    checkpoint_state = checkpoint['model_state_dict']
    model_state = model.state_dict()

    matched_keys = 0
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            matched_keys += 1
        else:
            skipped_keys.append(key)

    if matched_keys == 0:
        raise RuntimeError(
            "No matching parameters were found when loading the checkpoint. "
            "Ensure the SAM backbone and SigLIP configuration match the training setup."
        )

    model.load_state_dict(model_state)

    if skipped_keys:
        preview = ', '.join(skipped_keys[:5])
        if len(skipped_keys) > 5:
            preview += ', ...'
        print(
            f"Warning: skipped {len(skipped_keys)} checkpoint parameters due to mismatched shape or missing keys. "
            f"Examples: {preview}"
        )
    
    # Now transfer to GPU
    model = model.to(device)
    model.eval()
    
    return model

def make_prediction(model, image_tensor, text):
    """Make prediction using the CLIP-SAM model."""
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            output = model(image_tensor, text)
    return output 