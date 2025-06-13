#!/usr/bin/env python3
"""
Debug script to test stereo matching of Charuco boards.
"""

import os
import cv2
import numpy as np
import argparse
from charuco_detector import CharucoDetector

def main():
    parser = argparse.ArgumentParser(description='Debug stereo matching of Charuco boards')
    parser.add_argument('--left', type=str, required=True, help='Path to the left image file')
    parser.add_argument('--right', type=str, required=True, help='Path to the right image file')
    parser.add_argument('--squares_x', type=int, default=9, help='Number of squares in X direction')
    parser.add_argument('--squares_y', type=int, default=7, help='Number of squares in Y direction')
    parser.add_argument('--square_length', type=float, default=0.02, help='Length of square side in meters (20mm)')
    parser.add_argument('--marker_length', type=float, default=0.015, help='Length of marker side in meters (15mm)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    # Load images
    left_image = cv2.imread(args.left)
    right_image = cv2.imread(args.right)

    if left_image is None or right_image is None:
        print(f"Failed to load images: {args.left}, {args.right}")
        return

    print(f"Left image shape: {left_image.shape}")
    print(f"Right image shape: {right_image.shape}")

    # Create detector
    detector = CharucoDetector(
        args.squares_x, args.squares_y, args.square_length, args.marker_length
    )

    # Detect boards
    left_corners, left_ids, left_image_with_detections = detector.detect_board(left_image, draw=True)
    right_corners, right_ids, right_image_with_detections = detector.detect_board(right_image, draw=True)

    # Print detection results
    print("\nLeft image detection:")
    if left_corners is not None and left_ids is not None:
        print(f"Detected {len(left_corners)} Charuco corners with IDs: {left_ids.flatten()}")
    else:
        print("No Charuco corners detected")

    print("\nRight image detection:")
    if right_corners is not None and right_ids is not None:
        print(f"Detected {len(right_corners)} Charuco corners with IDs: {right_ids.flatten()}")
    else:
        print("No Charuco corners detected")

    # Save output images
    if left_image_with_detections is not None:
        cv2.imwrite(os.path.join(args.output_dir, 'debug_left.jpg'), left_image_with_detections)
    if right_image_with_detections is not None:
        cv2.imwrite(os.path.join(args.output_dir, 'debug_right.jpg'), right_image_with_detections)

    # Check for stereo matching
    if left_corners is not None and left_ids is not None and right_corners is not None and right_ids is not None:
        # Find common corners
        common_ids = np.intersect1d(left_ids, right_ids)

        print(f"\nCommon corner IDs: {common_ids}")
        print(f"Number of common corners: {len(common_ids)}")

        if len(common_ids) >= 4:
            print("Sufficient common corners for stereo calibration")

            # Get object points for the common corners
            object_points = []
            left_points = []
            right_points = []

            for id in common_ids:
                # Find index of the ID in left and right detections
                left_idx = np.where(left_ids == id)[0][0]
                right_idx = np.where(right_ids == id)[0][0]

                # Get the corner points
                left_point = left_corners[left_idx]
                right_point = right_corners[right_idx]

                # Get the object point (3D point on the board)
                # Note: getObjPoints() returns a tuple of arrays, not an array of points
                # We need to create the 3D point based on the board configuration
                # For a Charuco board, the points are in a grid with the specified square size
                row = id // args.squares_x
                col = id % args.squares_x
                object_point = np.array([col * args.square_length, row * args.square_length, 0], dtype=np.float32)

                # Add points to lists
                object_points.append(object_point)
                left_points.append(left_point)
                right_points.append(right_point)

                print(f"ID {id}: Left {left_point.flatten()}, Right {right_point.flatten()}")

            # Create a visualization of the matching
            combined_image = np.hstack((left_image_with_detections, right_image_with_detections))

            # Draw lines between matching points
            h, w = left_image.shape[:2]
            for i, id in enumerate(common_ids):
                left_idx = np.where(left_ids == id)[0][0]
                right_idx = np.where(right_ids == id)[0][0]

                left_pt = tuple(map(int, left_corners[left_idx].flatten()))
                right_pt = (int(right_corners[right_idx][0][0]) + w, int(right_corners[right_idx][0][1]))

                # Draw a line connecting the points
                cv2.line(combined_image, left_pt, right_pt, (0, 255, 255), 1)

                # Add ID text
                cv2.putText(combined_image, str(id), left_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(combined_image, str(id), right_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imwrite(os.path.join(args.output_dir, 'debug_stereo_matching.jpg'), combined_image)
            print(f"Stereo matching visualization saved to {os.path.join(args.output_dir, 'debug_stereo_matching.jpg')}")
        else:
            print("Insufficient common corners for stereo calibration (need at least 4)")
    else:
        print("Cannot perform stereo matching due to detection failures")

if __name__ == "__main__":
    main()
