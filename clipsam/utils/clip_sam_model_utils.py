import os
import torch
import sys

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def load_model(model_type, checkpoint_path, gpu_id=0):
    """Load the CLIP-SAM model from checkpoint."""
    if model_type != 'clip_sam':
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Import here to avoid circular imports
    from model import SigLipSamSegmentator
    
    # Initialize model with default parameters
    model = SigLipSamSegmentator(
        siglip_model_name='google/siglip2-so400m-patch14-384',
        sam_model_name='facebook/sam-vit-base',
        down_spatial_times=2,
        with_dense_feat=True,
        device='cpu'  # First load to CPU
    )
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load only trainable parameters
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items() 
        if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Now transfer to GPU
    model = model.to(f'cuda:{gpu_id}')
    model.eval()
    
    return model

def make_prediction(model, image_tensor, text):
    """Make prediction using the CLIP-SAM model."""
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            output = model(image_tensor, text)
    return output 