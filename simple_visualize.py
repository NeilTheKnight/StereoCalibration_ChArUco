#!/usr/bin/env python3
"""
Simple script to visualize the rectification results.
"""

import os
import cv2
import numpy as np
import json
import argparse

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
        if key != 'calibration_error' and key != 'baseline_mm' and key != 'known_baseline_mm' and key != 'board_config' and isinstance(params[key], list):
            params[key] = np.array(params[key])
    
    return params

def main():
    parser = argparse.ArgumentParser(description='Simple visualization of rectification results')
    parser.add_argument('--left', type=str, required=True, help='Path to the left image file')
    parser.add_argument('--right', type=str, required=True, help='Path to the right image file')
    parser.add_argument('--params', type=str, required=True, help='Path to the JSON file containing calibration parameters')
    parser.add_argument('--output', type=str, default='rectification_result.jpg', help='Output image file')
    
    args = parser.parse_args()
    
    # Load images
    left_image = cv2.imread(args.left)
    right_image = cv2.imread(args.right)
    
    if left_image is None or right_image is None:
        print(f"Failed to load images: {args.left}, {args.right}")
        return
    
    print(f"Left image shape: {left_image.shape}")
    print(f"Right image shape: {right_image.shape}")
    
    # Load calibration parameters
    params = load_calibration_params(args.params)
    
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
    
    # Draw horizontal lines for visual inspection of rectification
    line_interval = 50
    for y in range(0, h, line_interval):
        cv2.line(left_rectified, (0, y), (w-1, y), (0, 255, 0), 1)
        cv2.line(right_rectified, (0, y), (w-1, y), (0, 255, 0), 1)
    
    # Create side-by-side visualization
    vis = np.hstack((left_rectified, right_rectified))
    
    # Save visualization
    cv2.imwrite(args.output, vis)
    print(f"Rectification visualization saved to {args.output}")
    
    # Create anaglyph (red-cyan) 3D image
    if len(left_rectified.shape) == 3:
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_rectified
        right_gray = right_rectified
    
    # Create anaglyph (red channel from left image, blue and green from right)
    anaglyph = cv2.merge([left_gray, right_gray, right_gray])
    
    # Save anaglyph
    cv2.imwrite('anaglyph_' + args.output, anaglyph)
    print(f"Anaglyph image saved to anaglyph_{args.output}")

if __name__ == "__main__":
    main()
