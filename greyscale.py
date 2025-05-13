import cv2
import numpy as np
import os
from glob import glob

# Set your directory path
# mask_dir = "/home/minjilee/Desktop/may13/masks"
input_dir = "/home/minjilee/Desktop/may13/masks"
output_dir = "/home/minjilee/Desktop/may13/masks_grey"
os.makedirs(output_dir, exist_ok=True)

# Get all .png mask files
mask_files = glob(os.path.join(input_dir, "*.png"))

for mask_path in mask_files:
    # Load the RGB image
    mask_rgb = cv2.imread(mask_path)

    # Convert to grayscale by checking where pixels are not black (0,0,0)
    mask_binary = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_binary, 1, 255, cv2.THRESH_BINARY)

    # Save result
    base_name = os.path.basename(mask_path)
    output_path = os.path.join(output_dir, base_name)
    cv2.imwrite(output_path, mask_binary)

print("All masks converted to grayscale binary format.")