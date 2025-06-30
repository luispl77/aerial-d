import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw

def create_debug_visualization(image_path, xml_path, output_path):
    """Create debug visualization of the image with annotations"""
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Draw each object
    for obj in root.findall('object'):
        # Get the bounding box
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            
            # Add the class name
            name = obj.find('name').text
            draw.text((xmin, ymin - 15), name, fill="red")
        
        # If there's segmentation data, visualize it
        seg_elem = obj.find('segmentation')
        if seg_elem is not None:
            try:
                # Parse the RLE segmentation data
                seg_data = eval(seg_elem.text)
                if isinstance(seg_data, dict) and 'size' in seg_data and 'counts' in seg_data:
                    # Decode RLE to mask
                    rle = {'size': seg_data['size'], 'counts': seg_data['counts'].encode('utf-8')}
                    mask = mask_util.decode(rle)
                    
                    # Apply mask to image with semi-transparency
                    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    mask_rgb[mask == 1] = [255, 0, 0]  # Red color for the mask
                    
                    # Convert to PIL Image and apply with transparency
                    mask_image = Image.fromarray(mask_rgb)
                    image.paste(mask_image, (0, 0), mask=Image.fromarray((mask * 128).astype(np.uint8)))
            except:
                print(f"Error processing segmentation in {xml_path}")
    
    # Save the visualization
    image.save(output_path)

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches/annotations"  # Directory with XML files
    images_dir = "dataset/patches/images"  # Directory with patched images
    output_dir = "debug/patches/visualizations"  # Directory for visualizations
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    # Prepare paths for visualization
    xml_paths = []
    image_paths = []
    output_paths = []
    
    for xml_file in annotation_files:
        xml_path = os.path.join(input_dir, xml_file)
        
        # Get corresponding image filename from XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_filename = root.find('filename').text
        
        image_path = os.path.join(images_dir, image_filename)
        output_path = os.path.join(output_dir, f"vis_{os.path.splitext(image_filename)[0]}.png")
        
        xml_paths.append(xml_path)
        image_paths.append(image_path)
        output_paths.append(output_path)
    
    # Generate visualizations
    print(f"Generating visualizations for {len(annotation_files)} files...")
    for xml_path, image_path, output_path in tqdm(zip(xml_paths, image_paths, output_paths), 
                                                total=len(annotation_files),
                                                desc="Generating visualizations"):
        if os.path.exists(image_path):
            try:
                create_debug_visualization(image_path, xml_path, output_path)
            except Exception as e:
                print(f"Warning: Failed to create visualization for {os.path.basename(image_path)}: {e}")
        else:
            print(f"Warning: Image file not found: {image_path}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    main() 