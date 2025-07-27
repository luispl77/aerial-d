import os
import glob
import argparse
from flask import Flask, render_template, request, jsonify, send_file
import torch
from PIL import Image
import numpy as np
import json
from clip_sam_model_utils import load_model, make_prediction
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

app = Flask(__name__, static_folder='static')

# Get the workspace root directory (parent of utils)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration with absolute paths
DATASET_PATH = "/cfs/home/u035679/datasets/aeriald"
MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models')
VISUALIZATIONS_PATH = os.path.join(WORKSPACE_ROOT, 'utils', 'static', 'clip_sam', 'visualizations')

# Global variables
model = None
device = None
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-SAM Model Testing Interface')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for checkpoint loading')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--input_size', type=int, default=384, help='Input size for images')
    parser.add_argument('--get_domain_logits', action='store_true', help='Get domain classification logits during prediction')
    return parser.parse_args()

def get_isaid_images(split=None):
    """Get list of unique iSAID images in the dataset."""
    isaid_images = set()
    
    # If split is specified, only look in that split's directory
    splits = [split] if split else ['train', 'val']
    
    for current_split in splits:
        split_path = os.path.join(DATASET_PATH, current_split, 'images')
        if os.path.exists(split_path):
            for img_file in glob.glob(os.path.join(split_path, '*.png')):
                # Extract iSAID image number from filename
                filename = os.path.basename(img_file)
                isaid_num = filename.split('_')[0]  # P0000 from P0000_patch_000054
                isaid_images.add(isaid_num)
    
    return sorted(list(isaid_images))

def get_patches_for_isaid(isaid_num):
    """Get list of patches for a specific iSAID image."""
    patches = []
    
    for split in ['train', 'val']:
        split_path = os.path.join(DATASET_PATH, split, 'images')
        if os.path.exists(split_path):
            for img_file in glob.glob(os.path.join(split_path, f'{isaid_num}_*.png')):
                filename = os.path.basename(img_file)
                patches.append({
                    'path': f'/image/{split}/{filename}',  # Changed to use the new image route
                    'split': split,
                    'filename': filename
                })
    
    return sorted(patches, key=lambda x: x['filename'])

def preprocess_image(image_path, input_size=384):
    """Preprocess image for model input."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create transform
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(image)
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/patches/<isaid_num>')
def patches_page(isaid_num):
    """Render the patches page for a specific iSAID image."""
    return render_template('patches_clip_sam.html', isaid_num=isaid_num)

@app.route('/api/isaid_images')
def get_isaid_images_api():
    """Get list of iSAID images."""
    split = request.args.get('split', 'train')  # Default to train if not specified
    images = get_isaid_images(split)
    return jsonify(images)

@app.route('/api/patches/<isaid_num>')
def get_patches_api(isaid_num):
    """Get list of patches for a specific iSAID image."""
    patches = get_patches_for_isaid(isaid_num)
    return jsonify(patches)

@app.route('/image/<split>/<filename>')
def serve_image(split, filename):
    """Serve image files from the dataset."""
    image_path = os.path.join(DATASET_PATH, split, 'images', filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    return "Image not found", 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using selected model and image."""
    global model, device, args
    
    data = request.json
    image_path = data.get('image_path')
    text = data.get('text')
    
    if not image_path or not text:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Get initial VRAM usage
        initial_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
        
        # Convert the image path from URL to filesystem path
        split = image_path.split('/')[2]  # Get split from /image/train/filename.png
        filename = image_path.split('/')[-1]  # Get filename
        fs_image_path = os.path.join(DATASET_PATH, split, 'images', filename)
        
        # Preprocess image
        image_tensor = preprocess_image(fs_image_path, input_size=384)
        
        # Move to GPU
        image_tensor = image_tensor.to(device)
        
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start timing
        start_event.record()
        
        # Make prediction using the pre-loaded model
        domain_logits = None
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                # If domain logits are requested and the model supports it
                if args.get_domain_logits and hasattr(model, 'siglip_domain_classifier') and model.siglip_domain_classifier is not None:
                    output, domain_logits = model(image_tensor, text, return_domain_logits_inference=True)
                else:
                    # Standard prediction
                    output = model(image_tensor, text)

        # End timing and synchronize
        end_event.record()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        
        # Calculate prediction time in milliseconds
        prediction_time_ms = start_event.elapsed_time(end_event)
        
        # Get peak VRAM usage during prediction
        peak_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
        
        # Calculate total VRAM used for this prediction
        vram_used = peak_vram - initial_vram
        
        # Process domain logits if they exist
        domain_results = None
        if domain_logits is not None:
            domain_map = {0: 'iSAID', 1: 'DeepGlobe', 2: 'LoveDA'}
            domain_results = {}
            
            # Process SigLIP domain logits
            if 'siglip' in domain_logits:
                siglip_probs = torch.softmax(domain_logits['siglip'], dim=1).cpu().numpy()[0]
                domain_results['SigLIP'] = {name: f"{prob:.4f}" for name, prob in zip(domain_map.values(), siglip_probs)}
            
            # Process SAM domain logits
            if 'sam' in domain_logits:
                sam_probs = torch.softmax(domain_logits['sam'], dim=1).cpu().numpy()[0]
                domain_results['SAM'] = {name: f"{prob:.4f}" for name, prob in zip(domain_map.values(), sam_probs)}
            
            print(f"Domain Classification: {domain_results}")

        # Start visualization timing
        vis_start_time = time.time()
        
        # Convert prediction to probability map and move to CPU for visualization
        prob = torch.sigmoid(output[0]).cpu().numpy()
        
        # Create binary mask from probability map
        binary_mask = (prob > 0.5).astype(np.float32)
        
        # Load original image
        image = Image.open(fs_image_path)
        image_np = np.array(image)
        
        # Create visualization with three panels
        plt.figure(figsize=(30, 10))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot probability map
        plt.subplot(1, 3, 2)
        plt.imshow(prob, cmap='jet', vmin=0, vmax=1)
        plt.title(f'Prob Map (min={prob.min():.2f}, max={prob.max():.2f})')
        plt.axis('off')
        
        # Plot overlay with binary mask
        plt.subplot(1, 3, 3)
        plt.imshow(image_np)
        # Resize binary mask to match image dimensions
        mask_resized = np.array(Image.fromarray(binary_mask).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
        plt.imshow(mask_resized, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        plt.title('Overlay')
        plt.axis('off')
        
        # Add text as figure title
        plt.suptitle(f'Expression: "{text}"', wrap=True)
        
        # Save visualization with fixed filename based on patch filename
        os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
        vis_path = os.path.join(VISUALIZATIONS_PATH, f"{filename}_pred.png")
        
        # Save figure with tight layout
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Calculate visualization time
        visualization_time_ms = (time.time() - vis_start_time) * 1000
        
        # Measure VRAM after loading model
        model_vram = (torch.cuda.memory_reserved(device) - initial_vram) / 1024**2  # Convert to MB
        print(f"Model VRAM usage: {model_vram:.2f} MB")
        
        response_data = {
            'status': 'success',
            'result_path': f'/static/visualizations/{os.path.basename(vis_path)}',
            'metrics': {
                'prediction_time_ms': round(prediction_time_ms, 2),  # GPU time
                'visualization_time_ms': round(visualization_time_ms, 2),  # CPU time
                'total_time_ms': round(prediction_time_ms + visualization_time_ms, 2),
                'vram_used_mb': round(vram_used, 2),
                'peak_vram_mb': round(peak_vram, 2),
                'model_vram_mb': round(model_vram, 2)  # Add model VRAM usage
            }
        }
        
        if domain_results:
            response_data['domain_classification'] = domain_results
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization images."""
    response = send_file(os.path.join(VISUALIZATIONS_PATH, filename))
    # Add cache control headers to prevent browser caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Set up GPU
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    print(f"Using device: {device}")
    
    # Create checkpoint path
    checkpoint_path = os.path.join(MODEL_PATH, args.model_name, 'best.pt')
    
    # Measure VRAM before loading model
    initial_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(
        'clip_sam', 
        checkpoint_path, 
        args.gpu_id, 
        enable_domain_adaptation=args.get_domain_logits
    )
    model.eval()
    
    # Ensure model stays on the correct device
    model = model.to(device)
    
    # Measure VRAM after loading model
    model_vram = (torch.cuda.memory_reserved(device) - initial_vram) / 1024**2  # Convert to MB
    print(f"Model VRAM usage: {model_vram:.2f} MB")
    print(f"Model loaded successfully on {device}")
    
    # Create necessary directories
    os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, port=5001)
 