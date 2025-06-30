import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import pycocotools.mask as mask_util
import math
import argparse  # Add argparse for command line arguments

# ============================================================
# Configuration
# ============================================================
XML_INPUT_DIR = "dataset/patches_rules"  # Base directory for annotated XMLs
IMAGES_INPUT_DIR = "dataset/patches"     # Base directory for patch images
DEBUG_OUTPUT_DIR = "debug/patches_rules_color"  # Where to save debug images and logs
MAX_DEBUG_FILES = None  # Default to process all files
# ============================================================

# === Revised Color Constants ===
HUE_RANGES = {
    "red": ([0, 10], [170, 179]),
    "orange": ([11, 20],),
    "yellow": ([20, 34],),
    "green": ([35, 85],),
    "cyan": ([86, 100],),
    "blue": ([101, 130],),
    "purple": ([131, 145],),
    "magenta": ([146, 169],),
}
# HSV Ranges: H: 0-179, S: 0-255, V: 0-255
# Thresholds on a 0-100 scale for S and V
ACHROMATIC_SATURATION_THRESHOLD_100 = 25
ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100 = 54
MIN_PIXELS_FOR_COLOR = 100
# Dominance Thresholds (as fractions)
ACHROMATIC_DOMINANCE_THRESHOLD = 0.70
CHROMATIC_DOMINANCE_THRESHOLD = 0.70
SINGLE_HUE_MIN_PERC = 60.0
# ===============================

# === RLE Decode function (copied from previous scripts) ===
def rle_to_mask(rle_string, height, width):
    """Decode RLE string (stored as JSON) into a binary mask."""
    if not rle_string:
        return np.zeros((height, width), dtype=np.uint8)
    try:
        # Attempt to fix common JSON issues before parsing
        corrected_rle_string = rle_string.replace("'", '"')
        rle = json.loads(corrected_rle_string)

        # Ensure 'counts' is bytes for pycocotools
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('utf-8')

        # Check for size compatibility if possible
        if 'size' in rle and tuple(rle['size']) != (height, width):
             print(f"Warning: RLE size mismatch {rle['size']} vs target {(height, width)}. Using empty mask.")
             return np.zeros((height, width), dtype=np.uint8)

        mask = mask_util.decode(rle)

        # Final shape check after decoding
        if mask.shape != (height, width):
             print(f"Warning: RLE decode shape mismatch ({mask.shape} vs {(height, width)}). Using empty mask.")
             return np.zeros((height, width), dtype=np.uint8)
        return mask
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from RLE string: {e}")
        print(f"Problematic RLE string (start): {rle_string[:100]}...")
        return np.zeros((height, width), dtype=np.uint8)
    except Exception as e:
        print(f"Error decoding RLE: {e}. RLE String (start): '{rle_string[:100]}...'")
        return np.zeros((height, width), dtype=np.uint8)
# ======================================================== 

# === Format histogram data for debug output ===
def format_histogram(hist, max_bins=20, show_category_totals=True):
    """Format histogram data for debug output, showing the top bins and category totals."""
    if hist is None or np.sum(hist) == 0:
        return "No histogram data"
    
    # Flatten and normalize
    hist_flat = hist.flatten()
    total = np.sum(hist_flat)
    norm_hist = (hist_flat / total) * 100  # as percentage
    
    # Calculate totals by color category
    category_totals = {}
    for i in range(180):  # All possible hue values
        if norm_hist[i] > 0:  # Only consider non-zero bins
            # Map hue value to color name
            color_name = "unknown"
            for name, ranges in HUE_RANGES.items():
                for hue_range in ranges:
                    low, high = hue_range
                    if low > high:  # wrap-around
                        if i >= low or i <= high:
                            color_name = name
                            break
                    elif low <= i <= high:
                        color_name = name
                        break
                if color_name != "unknown":
                    break
            
            # Add to category total
            if color_name not in category_totals:
                category_totals[color_name] = 0
            category_totals[color_name] += norm_hist[i]
    
    # Format output for top individual bins
    # Get indices of top N bins
    top_indices = np.argsort(norm_hist)[::-1][:max_bins]
    
    bin_output = []
    for i in top_indices:
        if norm_hist[i] > 0:
            # Map hue value to color name
            color_name = "unknown"
            for name, ranges in HUE_RANGES.items():
                for hue_range in ranges:
                    low, high = hue_range
                    if low > high:  # wrap-around
                        if i >= low or i <= high:
                            color_name = name
                            break
                    elif low <= i <= high:
                        color_name = name
                        break
                if color_name != "unknown":
                    break
                    
            bin_output.append(f"H{i}({color_name}):{norm_hist[i]:.1f}%")
    
    # Format output for category totals
    category_output = []
    if show_category_totals and category_totals:
        category_output.append("\nCategory Totals: " + 
                             " ".join([f"{name}:{total:.1f}%" 
                                     for name, total in sorted(category_totals.items(), 
                                                             key=lambda x: x[1], reverse=True)]))
    
    # Combine outputs
    return " ".join(bin_output) + ("" if not category_output else category_output[0])
# ======================================================== 

# === Color Analysis function (Revised Logic with Category Grouping) ===
def determine_instance_color(instance_pixels_hsv):
    """Analyzes HSV pixels of an instance to determine its dominant color (light, dark, or specific hue)."""
    total_masked_pixels = instance_pixels_hsv.shape[0]
    determined_color = "None"
    analysis_details = ""
    histogram_data = None  # Store the histogram for debugging
    category_data = {}     # Store category percentages

    if total_masked_pixels == 0:
        return determined_color, "No Pixels in Array", total_masked_pixels, histogram_data, category_data

    if total_masked_pixels < MIN_PIXELS_FOR_COLOR:
        return determined_color, f"Too Few Pixels ({total_masked_pixels} < {MIN_PIXELS_FOR_COLOR})", total_masked_pixels, histogram_data, category_data

    # --- Extract H, S, V and scale S, V to 0-100 --- 
    h_values = instance_pixels_hsv[:, 0]
    s_values_100 = instance_pixels_hsv[:, 1] * (100.0 / 255.0)
    v_values_100 = instance_pixels_hsv[:, 2] * (100.0 / 255.0)
    # ------------------------------------------------

    # --- Classify each pixel --- 
    is_achromatic = s_values_100 < ACHROMATIC_SATURATION_THRESHOLD_100
    is_light = is_achromatic & (v_values_100 >= ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100)
    is_dark = is_achromatic & (v_values_100 < ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100)
    is_chromatic = ~is_achromatic
    # -------------------------

    # --- Calculate counts and percentages --- 
    light_pixels = np.sum(is_light)
    dark_pixels = np.sum(is_dark)
    chromatic_pixels = np.sum(is_chromatic)

    light_perc = (light_pixels / total_masked_pixels) * 100
    dark_perc = (dark_pixels / total_masked_pixels) * 100
    chromatic_perc = (chromatic_pixels / total_masked_pixels) * 100

    analysis_details = (f"Percentages: L:{light_perc:.1f}% D:{dark_perc:.1f}% C:{chromatic_perc:.1f}%")
    # --------------------------------------

    # --- Determine final color based on dominance --- 
    if light_perc >= ACHROMATIC_DOMINANCE_THRESHOLD * 100:
        determined_color = "light"
        analysis_details += f" | Light Dominant (>{ACHROMATIC_DOMINANCE_THRESHOLD*100}%)"
    elif dark_perc >= ACHROMATIC_DOMINANCE_THRESHOLD * 100:
        determined_color = "dark"
        analysis_details += f" | Dark Dominant (>{ACHROMATIC_DOMINANCE_THRESHOLD*100}%)"
    # Check if light wins and light+chromatic together exceed the threshold
    elif (light_perc + chromatic_perc) >= ACHROMATIC_DOMINANCE_THRESHOLD * 100 and light_perc > dark_perc and light_perc > chromatic_perc:
        determined_color = "light"
        analysis_details += f" | Light Wins and Light+Chromatic: {light_perc+chromatic_perc:.1f}% (L:{light_perc:.1f}% > C:{chromatic_perc:.1f}%) -> Light"
    # Check if dark wins and dark+chromatic together exceed the threshold
    elif (dark_perc + chromatic_perc) >= ACHROMATIC_DOMINANCE_THRESHOLD * 100 and dark_perc > light_perc and dark_perc > chromatic_perc:
        determined_color = "dark"
        analysis_details += f" | Dark Wins and Dark+Chromatic: {dark_perc+chromatic_perc:.1f}% (D:{dark_perc:.1f}% > C:{chromatic_perc:.1f}%) -> Dark"
    elif chromatic_perc >= CHROMATIC_DOMINANCE_THRESHOLD * 100:
        analysis_details += f" | Chromatic Dominant (>{CHROMATIC_DOMINANCE_THRESHOLD*100}%), Analyzing Hue..."
        
        if chromatic_pixels > 0:
            # --- NEW: Color Category Analysis ---
            chromatic_hues = h_values[is_chromatic]
            hist = cv2.calcHist([chromatic_hues], [0], None, [180], [0, 180])
            histogram_data = hist.copy()  # Save histogram for debug output
            total_chromatic_hist = np.sum(hist)
            
            if total_chromatic_hist > 0:
                # Group histogram bins by color category
                color_categories = {}
                for i in range(180):
                    bin_value = hist[i][0]
                    if bin_value > 0:  # Skip empty bins
                        # Find which color category this hue belongs to
                        category = None
                        for color_name, ranges in HUE_RANGES.items():
                            for hue_range in ranges:
                                low, high = hue_range
                                if low > high:  # wrap-around
                                    if i >= low or i <= high:
                                        category = color_name
                                        break
                                elif low <= i <= high:
                                    category = color_name
                                    break
                            if category:
                                break
                        
                        # Use "unknown" for any hue that doesn't match our ranges
                        category = category or "unknown"
                        
                        # Add to category total
                        if category not in color_categories:
                            color_categories[category] = 0
                        color_categories[category] += bin_value
                
                # Find the dominant color category and its percentage
                if color_categories:
                    dominant_category = max(color_categories, key=color_categories.get)
                    dominant_category_percentage = (color_categories[dominant_category] / total_chromatic_hist) * 100
                    
                    # Store category data for debugging
                    category_data = {k: (v / total_chromatic_hist) * 100 for k, v in color_categories.items()}
                    
                    # Include the dominant category and percentage in analysis
                    analysis_details += f" DomCategory:{dominant_category}@{dominant_category_percentage:.1f}%"
                    
                    # Check if dominant category exceeds threshold
                    if dominant_category_percentage >= SINGLE_HUE_MIN_PERC:
                        determined_color = dominant_category
                        analysis_details += f" | Category > {SINGLE_HUE_MIN_PERC}%, Mapped:{determined_color}"
                    else:
                        determined_color = "None"
                        analysis_details += f" | Category < {SINGLE_HUE_MIN_PERC}%, Too Mixed"
                else:
                    determined_color = "None"
                    analysis_details += " | No valid color categories found"
            else:
                determined_color = "None"
                analysis_details += " | No counts in chromatic hist"
        else:
            determined_color = "None"
            analysis_details += " | No chromatic pixels"
    else:
        determined_color = "None"
        analysis_details += " | No Dominant Category"

    analysis_details += f" | FinalCalc:{determined_color}"
    return determined_color, analysis_details, total_masked_pixels, histogram_data, category_data
# ============================================================

# === Visualization Function ===
def visualize_color_info(xml_path, image_path, output_image_path):
    """Calculates color, creates debug image showing ID, mask, calculated color, and detailed analysis log."""
    try:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Load image using PIL and prepare OpenCV version
        image_pil = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        hsv_image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV) # Convert full image once
        
        # Try to get a font
        try: 
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError: 
            font = ImageFont.load_default()
        
        color_debug_strings = []
        
        # First pass: Draw all masks at once on the OpenCV image
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            
            # Get segmentation RLE
            segmentation_elem = obj.find('segmentation')
            rle_string = segmentation_elem.text if segmentation_elem is not None else None
            
            if rle_string:
                mask = rle_to_mask(rle_string, height, width)
                mask_bool = mask.astype(bool)
                
                if np.any(mask_bool):
                    # Draw mask contour on OpenCV image
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # Use standard 1px line
                    cv2.drawContours(image_cv, contours, -1, (0, 255, 0), 1)
        
        # Convert the image with all masks back to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # Second pass: Calculate colors and add labels
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            debug_line_prefix = f"ID:{obj_id}"
            calculated_color = "None"
            analysis_details = "N/A"
            pixel_count = 0  # Initialize as integer instead of string
            mask_found = "No"
            histogram_data = None
            category_data = {}
            
            # Get segmentation RLE
            segmentation_elem = obj.find('segmentation')
            rle_string = segmentation_elem.text if segmentation_elem is not None else None
            
            if rle_string:
                mask = rle_to_mask(rle_string, height, width)
                mask_bool = mask.astype(bool)
                
                if np.any(mask_bool):
                    pixel_count = int(np.sum(mask_bool))
                    mask_found = "Yes"
                    
                    # --- Perform Color Calculation --- 
                    instance_pixels_hsv = hsv_image_cv[mask_bool]
                    calculated_color, analysis_details, _, histogram_data, category_data = determine_instance_color(instance_pixels_hsv)
                    # ---------------------------------
                else:
                    pixel_count = 0
                    mask_found = "Yes (Empty)"
                    analysis_details = "Empty Mask"
            else:
                analysis_details = "No Mask"

            # Append info to debug string
            # Make analysis_details more readable in the text file
            formatted_analysis = analysis_details.replace(" -> ", " | ")
            debug_line = f"{debug_line_prefix} Pixels:{pixel_count} Mask:{mask_found} | {formatted_analysis}"
            
            # Add histogram data if available and chromatic analysis was performed
            if histogram_data is not None and "Chromatic Dominant" in analysis_details:
                hist_str = format_histogram(histogram_data)
                debug_line += f"\nHistogram: {hist_str}"
                
                # Add category breakdown if available
                if category_data:
                    category_str = "\nCategory Percentages: " + " ".join([
                        f"{cat}:{perc:.1f}%" for cat, perc in 
                        sorted(category_data.items(), key=lambda x: x[1], reverse=True)
                    ])
                    debug_line += category_str
            
            color_debug_strings.append(debug_line)
            
            # --- Draw labels only (no bbox) --- 
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(bbox.find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]
                
                # Draw ID below the bbox (White with shadow for visibility)
                id_text = f"{obj_id}"
                shadow_offset = 1
                draw.text((xmin + shadow_offset, ymax + 5 + shadow_offset), id_text, fill=(0, 0, 0), font=font)
                draw.text((xmin, ymax + 5), id_text, fill=(255, 255, 255), font=font)
                
                # Draw Calculated Color above the bbox position
                if calculated_color != "None":
                    # Color exists - draw with bright yellow fill and shadow
                    draw.text((xmin + shadow_offset, ymin - 15 + shadow_offset), calculated_color, fill=(0, 0, 0), font=font)
                    draw.text((xmin, ymin - 15), calculated_color, fill=(255, 255, 0), font=font)
                elif pixel_count > 0:  # Only show "None" reason if there are pixels
                    # For None with pixels, provide minimal reason in gray
                    short_reason = "No Color" 
                    if "Too Few Pixels" in analysis_details:
                        short_reason = "Too Few"
                    elif "No Dominant Category" in analysis_details:
                        short_reason = "No Dominant"
                    elif "Category < " in analysis_details:
                        short_reason = "Mixed Hues"
                    elif "Hue Perc <" in analysis_details:  # For backward compatibility
                        short_reason = "Mixed Hues"
                    color_text = f"None ({short_reason})"
                    draw.text((xmin, ymin - 15), color_text, fill=(180, 180, 180), font=font)
        
        # Save the final image
        image_pil.save(output_image_path)
        return True, color_debug_strings
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return False, []
    except ET.ParseError:
        print(f"Error: Could not parse XML file {xml_path}")
        return False, []
    except Exception as e:
        print(f"Error creating visualization for {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False, []
# ============================

# === Main Execution Logic ===
def main():
    """Calculates color rules and generates debug visualizations and detailed text logs."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate color debug visualizations')
    parser.add_argument('--start', type=int, default=0, help='Starting file index (0-based)')
    parser.add_argument('--end', type=int, default=None, help='Ending file index (exclusive, None for all)')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--xml-dir', type=str, default=XML_INPUT_DIR, help='Base directory containing XML files')
    parser.add_argument('--img-dir', type=str, default=IMAGES_INPUT_DIR, help='Base directory containing images')
    parser.add_argument('--output-dir', type=str, default=DEBUG_OUTPUT_DIR, help='Output directory for debug files')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'all'], default='all', help='Which split to process')
    args = parser.parse_args()
    
    XML_DIR = args.xml_dir
    IMAGES_DIR = args.img_dir
    OUTPUT_DIR = args.output_dir
    
    print(f"Starting color calculation and debug generation...")
    print(f"XML Input: {XML_DIR}")
    print(f"Image Input: {IMAGES_DIR}")
    print(f"Debug Output: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine which splits to process
    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    
    total_processed = 0
    total_failed = 0
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Set up split-specific paths
        split_xml_dir = os.path.join(XML_DIR, split, 'annotations')
        split_images_dir = os.path.join(IMAGES_DIR, split, 'images')
        split_output_dir = os.path.join(OUTPUT_DIR, split)
        
        # Create split-specific output directory
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Get list of annotation files
        try:
            annotation_files = [f for f in os.listdir(split_xml_dir) if f.endswith('.xml')]
            annotation_files.sort()
        except FileNotFoundError:
            print(f"Error: XML input directory not found: {split_xml_dir}")
            continue
        
        # Limit number of files based on arguments
        total_files = len(annotation_files)
        start_idx = min(args.start, total_files)
        
        if args.end is not None:
            end_idx = min(args.end, total_files)
        else:
            end_idx = total_files
        
        if args.limit is not None:
            end_idx = min(start_idx + args.limit, end_idx)
        
        annotation_files = annotation_files[start_idx:end_idx]
        
        if not annotation_files:
            print(f"No XML files found to process in {split} split.")
            continue
            
        print(f"Processing {len(annotation_files)} files (range {start_idx} to {end_idx-1} out of {total_files})...")
        
        # Process files
        processed_count = 0
        failed_count = 0
        for xml_file in tqdm(annotation_files, desc=f"Generating Debug Info for {split}"):
            xml_path = os.path.join(split_xml_dir, xml_file)
            
            # Determine corresponding image and output paths
            try:
                # Need to peek into XML for the image filename
                tree = ET.parse(xml_path)
                root = tree.getroot()
                image_filename_elem = root.find('filename')
                if image_filename_elem is None or not image_filename_elem.text:
                     print(f"Warning: XML file {xml_file} missing filename. Skipping.")
                     failed_count += 1
                     continue
                image_filename = image_filename_elem.text
            except ET.ParseError:
                 print(f"Warning: Could not parse XML file {xml_file} to get image filename. Skipping.")
                 failed_count += 1
                 continue
                 
            image_path = os.path.join(split_images_dir, image_filename)
            base_name = os.path.splitext(image_filename)[0]
            output_image_path = os.path.join(split_output_dir, f"vis_{base_name}.png")
            output_txt_path = os.path.join(split_output_dir, f"vis_{base_name}.txt")
            
            # Check if the original image exists before attempting visualization
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}. Skipping visualization for {xml_file}.")
                failed_count += 1
                continue
                
            # Generate visualization (which now includes calculation) and get debug strings
            success, debug_strings = visualize_color_info(xml_path, image_path, output_image_path)
            
            if success:
                processed_count += 1
                # Write the debug strings to the text file
                try:
                    with open(output_txt_path, 'w') as f:
                        if debug_strings:
                            f.write("\n".join(debug_strings))
                        else:
                            f.write("No object information extracted.")
                except Exception as e:
                    print(f"Error writing debug text file {output_txt_path}: {e}")
                    failed_count += 1 # Count as failure if text write fails
            else:
                # Error message is printed within visualize_color_info
                failed_count += 1
                # Clean up potentially partially created image file if visualization failed
                if os.path.exists(output_image_path):
                    try:
                        os.remove(output_image_path)
                    except OSError:
                        pass # Ignore if removal fails
        
        total_processed += processed_count
        total_failed += failed_count
        print(f"\n{split.upper()} split complete:")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed/Skipped: {failed_count}")
    
    print(f"\nDebug generation complete for all splits.")
    print(f"Total successfully processed: {total_processed}")
    print(f"Total failed/skipped: {total_failed}")
    print(f"Debug files (images and detailed logs) saved to: {OUTPUT_DIR}")
# ============================

if __name__ == '__main__':
    main()