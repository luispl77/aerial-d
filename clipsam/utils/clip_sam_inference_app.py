import os
import argparse
from flask import Flask, render_template, request, jsonify, send_file
import torch
from PIL import Image
import numpy as np
import json
from clip_sam_model_utils import load_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import base64
import io
from werkzeug.utils import secure_filename
import uuid

# Initialize Flask app with proper template directory
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Get the workspace root directory (parent of utils)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration with absolute paths
MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models')
UPLOADS_PATH = os.path.join(WORKSPACE_ROOT, 'utils', 'static', 'uploads')
RESULTS_PATH = os.path.join(WORKSPACE_ROOT, 'utils', 'static', 'results')

# Global variables
model = None
device = None
input_size = 384

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-SAM Interactive Inference Interface')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for checkpoint loading')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--input_size', type=int, default=384, help='Input size for images')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the Flask app on')
    return parser.parse_args()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, input_size=384):
    """Preprocess image for model input using black bar strategy (letterboxing/pillarboxing)."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Calculate scale factor to fit image within target size while maintaining aspect ratio
    scale = min(input_size / original_size[0], input_size / original_size[1])
    
    # Calculate new size
    new_width = int(original_size[0] * scale)
    new_height = int(original_size[1] * scale)
    
    # Resize image maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with target size and black background
    letterboxed_image = Image.new('RGB', (input_size, input_size), (0, 0, 0))
    
    # Calculate position to paste the resized image (center it)
    paste_x = (input_size - new_width) // 2
    paste_y = (input_size - new_height) // 2
    
    # Paste the resized image onto the black background
    letterboxed_image.paste(image, (paste_x, paste_y))
    
    # Create transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(letterboxed_image)
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

def process_base64_image(base64_string):
    """Process base64 image data and save to uploads directory."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Create PIL image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Generate unique filename
        filename = f"clipboard_{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOADS_PATH, filename)
        
        # Save image with high quality
        image.save(filepath, 'PNG', optimize=False)
        
        return filename, filepath
    except Exception as e:
        raise ValueError(f"Failed to process clipboard image: {str(e)}")

def calculate_letterbox_params(original_size, target_size):
    """Calculate letterboxing parameters for proper mask overlay."""
    original_w, original_h = original_size
    target_w, target_h = target_size, target_size
    
    # Calculate scale factor
    scale = min(target_w / original_w, target_h / original_h)
    
    # Calculate new dimensions
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    return {
        'scale': scale,
        'new_size': (new_w, new_h),
        'padding': (pad_x, pad_y),
        'crop_box': (pad_x, pad_y, pad_x + new_w, pad_y + new_h)
    }

@app.route('/')
def index():
    """Render the main inference page."""
    return render_template('inference.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(UPLOADS_PATH, filename)
            
            # Save file
            file.save(filepath)
            
            return jsonify({
                'status': 'success',
                'filename': filename,
                'url': f'/static/uploads/{filename}'
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/paste', methods=['POST'])
def paste_image():
    """Handle clipboard image paste."""
    try:
        data = request.json
        base64_string = data.get('image')
        
        if not base64_string:
            return jsonify({'error': 'No image data provided'}), 400
        
        filename, filepath = process_base64_image(base64_string)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'url': f'/static/uploads/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using the loaded model."""
    global model, device
    
    data = request.json
    filename = data.get('filename')
    text = data.get('text')
    
    if not filename or not text:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Get initial VRAM usage
        initial_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
        
        # Get image path
        image_path = os.path.join(UPLOADS_PATH, filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Preprocess image
        image_tensor = preprocess_image(image_path, input_size=input_size)
        
        # Move to GPU
        image_tensor = image_tensor.to(device)
        
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start timing
        start_event.record()
        
        # Make prediction using the pre-loaded model
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                # Ensure model is on GPU
                model = model.to(device)
                output = model(image_tensor, [text])  # Pass text as list
        
        # End timing and synchronize
        end_event.record()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        
        # Calculate prediction time in milliseconds
        prediction_time_ms = start_event.elapsed_time(end_event)
        
        # Get peak VRAM usage during prediction
        peak_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
        
        # Calculate total VRAM used for this prediction
        vram_used = peak_vram - initial_vram
        
        # Start visualization timing
        vis_start_time = time.time()
        
        # Convert prediction to probability map and move to CPU for visualization
        prob = torch.sigmoid(output[0]).cpu().numpy()
        
        # Create binary mask from probability map
        binary_mask = (prob > 0.5).astype(np.float32)
        
        # Load original image
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # Calculate letterboxing parameters for proper mask handling
        letterbox_params = calculate_letterbox_params(image.size, input_size)
        
        # Extract the relevant part of the probability map (remove black bars)
        prob_cropped = prob[letterbox_params['crop_box'][1]:letterbox_params['crop_box'][3],
                           letterbox_params['crop_box'][0]:letterbox_params['crop_box'][2]]
        
        # Create binary mask from cropped probability map
        binary_mask_cropped = (prob_cropped > 0.5).astype(np.float32)
        
        # Create visualization with three panels
        plt.figure(figsize=(30, 10))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot probability map (cropped to remove black bars)
        plt.subplot(1, 3, 2)
        plt.imshow(prob_cropped, cmap='jet', vmin=0, vmax=1)
        plt.title(f'Probability Map (min={prob_cropped.min():.3f}, max={prob_cropped.max():.3f})')
        plt.axis('off')
        
        # Plot overlay with properly resized binary mask
        plt.subplot(1, 3, 3)
        plt.imshow(image_np)
        # Resize cropped binary mask to match original image dimensions
        mask_resized = np.array(Image.fromarray(binary_mask_cropped).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
        plt.imshow(mask_resized, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        plt.title('Segmentation Overlay')
        plt.axis('off')
        
        # Add text as figure title
        plt.suptitle(f'Referring Expression: "{text}"', fontsize=16, wrap=True)
        
        # Save visualization
        result_filename = f"result_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(RESULTS_PATH, result_filename)
        
        # Save figure with tight layout
        plt.tight_layout()
        plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close()
        
        # Calculate visualization time
        visualization_time_ms = (time.time() - vis_start_time) * 1000
        
        # Calculate statistics using cropped data (excluding black bars)
        mask_coverage = float(np.mean(binary_mask_cropped) * 100)  # Convert to Python float
        confidence_avg = float(np.mean(prob_cropped))  # Convert to Python float
        confidence_max = float(np.max(prob_cropped))   # Convert to Python float
        
        return jsonify({
            'status': 'success',
            'result_url': f'/static/results/{result_filename}',
            'metrics': {
                'prediction_time_ms': round(float(prediction_time_ms), 2),
                'visualization_time_ms': round(visualization_time_ms, 2),
                'total_time_ms': round(float(prediction_time_ms) + visualization_time_ms, 2),
                'vram_used_mb': round(vram_used, 2),
                'peak_vram_mb': round(peak_vram, 2),
                'mask_coverage_percent': round(mask_coverage, 2),
                'avg_confidence': round(confidence_avg, 3),
                'max_confidence': round(confidence_max, 3)
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded images."""
    return send_file(os.path.join(UPLOADS_PATH, filename))

@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve result images."""
    response = send_file(os.path.join(RESULTS_PATH, filename))
    # Add cache control headers to prevent browser caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/clear_uploads', methods=['POST'])
def clear_uploads():
    """Clear uploaded files."""
    try:
        # Clear uploads directory
        for filename in os.listdir(UPLOADS_PATH):
            file_path = os.path.join(UPLOADS_PATH, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Clear results directory
        for filename in os.listdir(RESULTS_PATH):
            file_path = os.path.join(RESULTS_PATH, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        return jsonify({'status': 'success', 'message': 'All files cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Set up GPU
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    input_size = args.input_size
    print(f"Using device: {device}")
    
    # Create checkpoint path
    checkpoint_path = os.path.join(MODEL_PATH, args.model_name, 'best.pt')
    
    # Measure VRAM before loading model
    initial_vram = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model('clip_sam', checkpoint_path, args.gpu_id)
    model.eval()
    
    # Measure VRAM after loading model
    model_vram = (torch.cuda.memory_reserved(device) - initial_vram) / 1024**2  # Convert to MB
    print(f"Model VRAM usage: {model_vram:.2f} MB")
    
    # Create necessary directories
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print(f"Starting inference app on port {args.port}")
    print(f"Access the app at: http://localhost:{args.port}")
    
    # Run Flask app
    app.run(debug=True, port=args.port, host='0.0.0.0') 