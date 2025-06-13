#!/usr/bin/env python3
"""
Debug script to test Charuco board detection in images.
"""

import os
import cv2
import numpy as np
import argparse
from charuco_detector import CharucoDetector

def main():
    parser = argparse.ArgumentParser(description='Debug Charuco board detection in images')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--squares_x', type=int, default=9, help='Number of squares in X direction')
    parser.add_argument('--squares_y', type=int, default=7, help='Number of squares in Y direction')
    parser.add_argument('--square_length', type=float, default=0.02, help='Length of square side in meters (20mm)')
    parser.add_argument('--marker_length', type=float, default=0.015, help='Length of marker side in meters (15mm)')
    parser.add_argument('--output', type=str, default='debug_output.jpg', help='Output image file')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Create detector
    detector = CharucoDetector(
        args.squares_x, args.squares_y, args.square_length, args.marker_length
    )
    
    # Detect board
    corners, ids, image_with_detections = detector.detect_board(image, draw=True)
    
    # Print detection results
    if corners is not None and ids is not None:
        print(f"Detected {len(corners)} Charuco corners with IDs: {ids.flatten()}")
    else:
        print("No Charuco corners detected")
    
    # Save output image
    if image_with_detections is not None:
        cv2.imwrite(args.output, image_with_detections)
        print(f"Debug image saved to {args.output}")
    
    # Try with different parameters
    print("\nTrying with different parameters...")
    
    # Try with different dictionary
    print("\nTrying with DICT_4X4_250 dictionary...")
    detector = CharucoDetector(
        args.squares_x, args.squares_y, args.square_length, args.marker_length,
        dictionary_id=cv2.aruco.DICT_4X4_250
    )
    corners, ids, image_with_detections = detector.detect_board(image, draw=True)
    if corners is not None and ids is not None:
        print(f"Detected {len(corners)} Charuco corners with IDs: {ids.flatten()}")
        cv2.imwrite('debug_dict_4x4.jpg', image_with_detections)
    else:
        print("No Charuco corners detected")
    
    # Try with different square sizes
    print("\nTrying with different square sizes...")
    for square_length, marker_length in [(0.04, 0.03), (0.03, 0.02), (0.025, 0.018)]:
        print(f"\nTrying with square_length={square_length}m, marker_length={marker_length}m")
        detector = CharucoDetector(
            args.squares_x, args.squares_y, square_length, marker_length
        )
        corners, ids, image_with_detections = detector.detect_board(image, draw=True)
        if corners is not None and ids is not None:
            print(f"Detected {len(corners)} Charuco corners with IDs: {ids.flatten()}")
            cv2.imwrite(f'debug_square_{int(square_length*1000)}mm.jpg', image_with_detections)
        else:
            print("No Charuco corners detected")
    
    # Try with different board sizes
    print("\nTrying with different board sizes...")
    for squares_x, squares_y in [(7, 5), (10, 8), (8, 6)]:
        print(f"\nTrying with squares_x={squares_x}, squares_y={squares_y}")
        detector = CharucoDetector(
            squares_x, squares_y, args.square_length, args.marker_length
        )
        corners, ids, image_with_detections = detector.detect_board(image, draw=True)
        if corners is not None and ids is not None:
            print(f"Detected {len(corners)} Charuco corners with IDs: {ids.flatten()}")
            cv2.imwrite(f'debug_board_{squares_x}x{squares_y}.jpg', image_with_detections)
        else:
            print("No Charuco corners detected")

if __name__ == "__main__":
    main()
