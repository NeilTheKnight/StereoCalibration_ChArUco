#!/usr/bin/env python3
"""
Visualize rectification of stereo images using calibration results.
"""

import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_calibration_params(params_file):
    """
    Load calibration parameters from a JSON file.
    
    Args:
        params_file (str): Path to the JSON file containing calibration parameters
        
    Returns:
        dict: Calibration parameters
    """
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Convert lists back to numpy arrays
    for key in params:
        if key != 'calibration_error' and isinstance(params[key], list):
            params[key] = np.array(params[key])
    
    return params

def rectify_stereo_images(left_image, right_image, params):
    """
    Rectify stereo images using calibration parameters.
    
    Args:
        left_image (numpy.ndarray): Left camera image
        right_image (numpy.ndarray): Right camera image
        params (dict): Calibration parameters
        
    Returns:
        tuple: (left_rectified, right_rectified)
            left_rectified: Rectified left image
            right_rectified: Rectified right image
    """
    # Get image size
    h, w = left_image.shape[:2]
    
    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        params['left_camera_matrix'], params['left_dist_coeffs'], 
        params['left_rectification_matrix'], params['left_projection_matrix'], 
        (w, h), cv2.CV_32FC1)
    
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        params['right_camera_matrix'], params['right_dist_coeffs'], 
        params['right_rectification_matrix'], params['right_projection_matrix'], 
        (w, h), cv2.CV_32FC1)
    
    # Rectify images
    left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified

def create_anaglyph(left_image, right_image):
    """
    Create an anaglyph (red-cyan) 3D image from stereo pair.
    
    Args:
        left_image (numpy.ndarray): Left camera image
        right_image (numpy.ndarray): Right camera image
        
    Returns:
        numpy.ndarray: Anaglyph image
    """
    # Convert to grayscale if needed
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image
    
    # Create color channels
    zeros = np.zeros_like(left_gray)
    
    # Create anaglyph (red channel from left image, blue and green from right)
    anaglyph = cv2.merge([left_gray, right_gray, right_gray])
    
    return anaglyph

def main():
    parser = argparse.ArgumentParser(description='Visualize rectification of stereo images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing left and right subdirectories with stereo image pairs')
    parser.add_argument('--params', type=str, required=True, help='Path to the JSON file containing calibration parameters')
    parser.add_argument('--output_dir', type=str, default='rectification_results', help='Output directory for rectified images')
    parser.add_argument('--anaglyph', action='store_true', help='Create anaglyph (red-cyan) 3D images')
    
    args = parser.parse_args()
    
    # Check if input directory exists and has left and right subdirectories
    left_dir = os.path.join(args.input_dir, 'left')
    right_dir = os.path.join(args.input_dir, 'right')
    
    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        print(f"Error: Input directory must contain 'left' and 'right' subdirectories")
        return
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'left_rectified'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'right_rectified'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'side_by_side'), exist_ok=True)
    
    if args.anaglyph:
        os.makedirs(os.path.join(args.output_dir, 'anaglyph'), exist_ok=True)
    
    # Load calibration parameters
    params = load_calibration_params(args.params)
    
    # Get all image files from left directory
    left_files = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Get all image files from right directory
    right_files = sorted([f for f in os.listdir(right_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Find common filenames
    common_files = set(left_files).intersection(set(right_files))
    
    # Process each image pair
    for filename in common_files:
        print(f"Processing {filename}...")
        
        # Load images
        left_image = cv2.imread(os.path.join(left_dir, filename))
        right_image = cv2.imread(os.path.join(right_dir, filename))
        
        if left_image is None or right_image is None:
            print(f"Failed to load image pair: {filename}")
            continue
        
        # Rectify images
        left_rectified, right_rectified = rectify_stereo_images(left_image, right_image, params)
        
        # Save rectified images
        cv2.imwrite(os.path.join(args.output_dir, 'left_rectified', filename), left_rectified)
        cv2.imwrite(os.path.join(args.output_dir, 'right_rectified', filename), right_rectified)
        
        # Create and save side-by-side visualization
        h, w = left_rectified.shape[:2]
        
        # Draw horizontal lines for visual inspection of rectification
        line_interval = 50
        left_lined = left_rectified.copy()
        right_lined = right_rectified.copy()
        
        for y in range(0, h, line_interval):
            cv2.line(left_lined, (0, y), (w-1, y), (0, 255, 0), 1)
            cv2.line(right_lined, (0, y), (w-1, y), (0, 255, 0), 1)
        
        side_by_side = np.hstack((left_lined, right_lined))
        cv2.imwrite(os.path.join(args.output_dir, 'side_by_side', filename), side_by_side)
        
        # Create and save anaglyph if requested
        if args.anaglyph:
            anaglyph = create_anaglyph(left_rectified, right_rectified)
            cv2.imwrite(os.path.join(args.output_dir, 'anaglyph', filename), anaglyph)
    
    print(f"Rectification visualization completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
