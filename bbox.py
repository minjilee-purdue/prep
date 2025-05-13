import cv2
import numpy as np
import os
import json
from glob import glob
from collections import Counter

# Define directory paths
image_dir = "/home/minjilee/Desktop/may13/images"
mask_dir = "/home/minjilee/Desktop/may13/masks"
output_dir_jpg = "/home/minjilee/Desktop/may13/bbox_jpg"
output_dir_json = "/home/minjilee/Desktop/may13/coco_json"
output_dir_txt = "/home/minjilee/Desktop/may13/bbox_txt"  # Directory for human-readable text files

# Create output directories if they don't exist
os.makedirs(output_dir_jpg, exist_ok=True)
os.makedirs(output_dir_json, exist_ok=True)
os.makedirs(output_dir_txt, exist_ok=True)  # Create txt directory

# Define segmentation colors with hex values
color_map = {
    "Layer 1 (Background)": "#000000",
    "Layer 2": "#ffffff",
    "Layer 3": "#ff0000",
    "Layer 4": "#00ff00",
    "Layer 5": "#0000ff",
    "Layer 6": "#ff00cc",
    "Layer 7": "#ff6600",
    "Layer 8": "#ffcc00",
    "Layer 9": "#00ffcc",
    "Layer 10": "#0088ff",
    "Layer 11": "#ddffdc",
}

# Convert hex colors to RGB tuples
color_map_rgb = {}
for layer, hex_color in color_map.items():
    # Skip the # and convert each pair of hex digits to decimal
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    color_map_rgb[layer] = (r, g, b)

# Function to check if a color approximately matches any in our color map
def find_closest_color(pixel_color, color_map_rgb, tolerance=10):
    for layer, rgb_color in color_map_rgb.items():
        # Check if color is within tolerance range
        if (abs(pixel_color[0] - rgb_color[0]) <= tolerance and
            abs(pixel_color[1] - rgb_color[1]) <= tolerance and
            abs(pixel_color[2] - rgb_color[2]) <= tolerance):
            return layer
    return None

# Function to analyze mask and check for invalid colors
def validate_mask_colors(mask_rgb, color_map_rgb, tolerance=10, max_pixels_to_check=1000):
    height, width = mask_rgb.shape[:2]
    
    # Sample a subset of pixels for efficiency
    step_h = max(1, height // int(np.sqrt(max_pixels_to_check)))
    step_w = max(1, width // int(np.sqrt(max_pixels_to_check)))
    
    # Count occurrences of each unique color
    color_counts = Counter()
    invalid_colors = []
    
    for y in range(0, height, step_h):
        for x in range(0, width, step_w):
            pixel_color = tuple(mask_rgb[y, x])
            
            # Skip black background pixels to reduce noise
            if pixel_color == (0, 0, 0):
                continue
                
            # Check if this pixel color matches any in our color map
            matched_layer = find_closest_color(pixel_color, color_map_rgb, tolerance)
            
            if matched_layer:
                color_counts[matched_layer] += 1
            else:
                # If no match, add to invalid colors
                color_counts["invalid"] += 1
                if len(invalid_colors) < 10:  # Limit to avoid too many
                    invalid_colors.append(pixel_color)
    
    return color_counts, invalid_colors

# Create category map for COCO format
category_map = {}  # Map from layer name to category_id
categories = []
category_id = 1  # Start category_id from 1 (COCO standard)

for layer in color_map.keys():
    if layer == "Layer 1 (Background)":
        continue  # Skip background layer
    category_map[layer] = category_id
    categories.append({
        "id": category_id,
        "name": layer,
        "supercategory": "object"
    })
    category_id += 1

# Get all image files
image_files = sorted(glob(os.path.join(image_dir, "*.jpg"))) + sorted(glob(os.path.join(image_dir, "*.png")))
mask_files = sorted(glob(os.path.join(mask_dir, "*.jpg"))) + sorted(glob(os.path.join(mask_dir, "*.png")))

if len(image_files) != len(mask_files):
    print(f"WARNING: Number of image files ({len(image_files)}) doesn't match number of mask files ({len(mask_files)})")

# Color tolerance for matching
color_tolerance = 10

# Process each image
for i, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
    # Get base filename without extension
    image_filename = os.path.basename(image_path)
    base_filename = os.path.splitext(image_filename)[0]
    
    '''
    print(f"\nProcessing image {i+1}/{len(image_files)}: {image_filename}")
    '''
    # Define output paths for this image
    output_image_path = os.path.join(output_dir_jpg, f"{base_filename}_bbox.jpg")
    json_output_path = os.path.join(output_dir_json, f"{base_filename}.json")
    txt_output_path = os.path.join(output_dir_txt, f"{base_filename}_bboxes.txt")  # Human-readable text file
    
    # Load images
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}. Skipping.")
        continue
        
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"ERROR: Could not load mask from {mask_path}. Skipping.")
        continue

    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create a COCO dataset structure for this image
    coco_image = {
        "info": {
            "description": f"Bounding Box Data for {image_filename}",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "MinjiLee",
            "date_created": "2025-05-13"
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": [{
            "id": 1,  # Always 1 since this is a single-image file
            "license": 1,
            "file_name": image_filename,
            "height": height,
            "width": width,
            "date_captured": "2025-05-13"
        }],
        "annotations": [],
        "categories": categories  # Always include ALL possible categories
    }

    # Convert to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    
    # Validate mask colors
    '''
    print(f"Validating colors in mask: {os.path.basename(mask_path)}")
    '''
    color_counts, invalid_colors = validate_mask_colors(mask_rgb, color_map_rgb, color_tolerance)
    
    # Track which layers are actually present in this image
    detected_layers = set()
    
    '''
    # Report on colors found
    print("Colors detected in mask:")
    for layer, count in color_counts.items():
        if layer != "invalid" and layer != "Layer 1 (Background)":
            print(f"  - {layer}: {count} pixels")
            detected_layers.add(layer)
    '''

    '''
    # Report on number of layers detected
    print(f"Total layers detected: {len(detected_layers)} out of {len(categories)}")
    '''

    '''
    # Report on invalid colors
    if "invalid" in color_counts and color_counts["invalid"] > 0:
        print(f"WARNING: Found {color_counts['invalid']} pixels with colors that don't match any defined layer!")
        print("Examples of invalid colors (RGB):")
        for color in invalid_colors:
            print(f"  - RGB{color}")
        print(f"Check if your mask has colors not defined in the color map or if the color tolerance ({color_tolerance}) is too low.")\'
    '''

    # Create output image copy
    output_image = image_rgb.copy()

    # Initialize annotation counter for this image
    annotation_id = 1

    # Dictionary to store bounding boxes by layer
    layer_bboxes = {}
    
    # Initialize all layers with empty lists in the dictionary
    for layer in color_map.keys():
        if layer != "Layer 1 (Background)":
            layer_bboxes[layer] = []

    # Process each color separately
    for layer, rgb_color in color_map_rgb.items():
        if layer == "Layer 1 (Background)":
            continue  # Skip background
        
        # Create lower and upper bounds for color matching
        lower_bound = np.array([max(0, c - color_tolerance) for c in rgb_color], dtype=np.uint8)
        upper_bound = np.array([min(255, c + color_tolerance) for c in rgb_color], dtype=np.uint8)
        
        # Create a binary mask for the current color
        color_mask = cv2.inRange(mask_rgb, lower_bound, upper_bound)
        
        # Find contours for the current color mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours were found for this color
        if not contours:
            # This layer is not present in this image
            continue
            
        # Draw bounding boxes and store annotations
        valid_contours = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Only process contours with reasonable size
            if w > 5 and h > 5:  # Filtering small noise
                # Calculate coordinates
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                
                # Draw rectangle on output image
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), rgb_color, thickness=2)
                
                # Add to layer bboxes dictionary (in xmin, ymin, xmax, ymax format for human readability)
                layer_bboxes[layer].append((xmin, ymin, xmax, ymax))
                
                # Add annotation to COCO dataset for this image (in x, y, w, h format for COCO)
                coco_image["annotations"].append({
                    "id": annotation_id,
                    "image_id": 1,  # Always 1 since this is a single-image file
                    "category_id": category_map[layer],
                    "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
                    "area": w * h,
                    "segmentation": [],  # Empty for now, we only have bounding boxes
                    "iscrowd": 0
                })
                annotation_id += 1
                valid_contours += 1
        
        
        # if valid_contours > 0:
        #     '''
        #     print(f"  - Found {len(contours)} contours for {layer}, {valid_contours} valid bounding boxes")
        #     ''''
            
    # Save the output image with bounding boxes
    cv2.imwrite(output_image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Saved output image: {os.path.basename(output_image_path)}")
    
    # Save COCO annotations to individual JSON file for this image
    with open(json_output_path, 'w') as f:
        json.dump(coco_image, f, indent=2)
    print(f"Saved COCO annotations to: {os.path.basename(json_output_path)}")

    # Save bounding box coordinates to a human-readable text file
    # Only include layers that have at least one bounding box
    with open(txt_output_path, 'w') as f:
        # If no bounding boxes were found, write a message
        if sum(len(bboxes) for bboxes in layer_bboxes.values()) == 0:
            f.write(f"No valid bounding boxes found in {image_filename}\n")
        else:
            # Write each layer's bounding boxes
            for layer, bboxes in layer_bboxes.items():
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox
                    f.write(f"{layer}: {xmin}, {ymin}, {xmax}, {ymax}\n")
    
    print(f"Saved human-readable bounding boxes to: {os.path.basename(txt_output_path)}")

# Also create a combined JSON file that includes all images
print("\nCreating combined COCO JSON file...")
combined_coco = {
    "info": {
        "description": "Combined Bounding Box Dataset",
        "url": "",
        "version": "1.0",
        "year": 2025,
        "contributor": "MinjiLee",
        "date_created": "2025-05-13"
    },
    "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
    "images": [],
    "annotations": [],
    "categories": categories  # Always include ALL possible categories
}

# Re-process to create a combined file
image_id = 1
annotation_id = 1

for i, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
    # Get base filename without extension
    image_filename = os.path.basename(image_path)
    
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    height, width = image.shape[:2]
    
    # Add image to combined dataset
    combined_coco["images"].append({
        "id": image_id,
        "license": 1,
        "file_name": image_filename,
        "height": height,
        "width": width,
        "date_captured": "2025-05-13"
    })
    
    # Load individual JSON file to get annotations
    base_filename = os.path.splitext(image_filename)[0]
    json_path = os.path.join(output_dir_json, f"{base_filename}.json")
    
    try:
        with open(json_path, 'r') as f:
            image_data = json.load(f)
            
        # Add annotations with updated IDs
        for ann in image_data["annotations"]:
            ann_copy = ann.copy()
            ann_copy["id"] = annotation_id
            ann_copy["image_id"] = image_id
            combined_coco["annotations"].append(ann_copy)
            annotation_id += 1
    except Exception as e:
        print(f"Warning: Could not load or process JSON for {image_filename}: {e}")
    
    image_id += 1

# Save combined COCO file
combined_path = os.path.join(output_dir_json, "annotations.json")
with open(combined_path, 'w') as f:
    json.dump(combined_coco, f, indent=2)
print(f"Saved combined COCO annotations to: annotations.json")

# Create a dataset summary
print("\nCreating dataset summary...")
dataset_summary = {
    "total_images": len(image_files),
    "total_annotations": annotation_id - 1,
    "categories": [cat["name"] for cat in categories],
    "category_counts": {}
}

# Count annotations per category
for ann in combined_coco["annotations"]:
    cat_id = ann["category_id"]
    cat_name = next(cat["name"] for cat in categories if cat["id"] == cat_id)
    if cat_name not in dataset_summary["category_counts"]:
        dataset_summary["category_counts"][cat_name] = 0
    dataset_summary["category_counts"][cat_name] += 1

# Save dataset summary
summary_path = os.path.join(output_dir_json, "dataset_summary.json")
with open(summary_path, 'w') as f:
    json.dump(dataset_summary, f, indent=2)
print(f"Saved dataset summary to: dataset_summary.json")

print("\nProcessing complete")