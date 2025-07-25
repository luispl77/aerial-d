import os
import torch
import sys

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def load_model(model_type, checkpoint_path, gpu_id=0, enable_domain_adaptation=False):
    """Load the CLIP-SAM model from checkpoint."""
    if model_type != 'clip_sam':
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Import here to avoid circular imports
    from model import SigLipSamSegmentator
    
    # Initialize model with default parameters, considering domain adaptation
    model = SigLipSamSegmentator(
        siglip_model_name='google/siglip2-so400m-patch14-384',
        sam_model_name='facebook/sam-vit-base',
        down_spatial_times=2,
        with_dense_feat=True,
        device='cpu',  # First load to CPU
        enable_domain_adaptation=enable_domain_adaptation
    )
    
    # Load checkpoint to CPU first
    device = torch.device(f'cuda:{gpu_id}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load parameters, ignoring missing keys (e.g., domain classifier in non-DA model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
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