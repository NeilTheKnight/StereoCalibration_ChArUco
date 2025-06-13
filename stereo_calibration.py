#!/usr/bin/env python3
"""
Stereo camera calibration using Charuco boards.
"""

import os
import cv2
import numpy as np
from charuco_detector import CharucoDetector
from camera_calibration import CameraCalibrator

class StereoCalibrator:
    def __init__(self, detector=None, squares_x=9, squares_y=7, square_length=0.02, marker_length=0.015,
                dictionary_id=cv2.aruco.DICT_6X6_250, baseline=0.1):
        """
        Initialize the stereo calibrator.

        Args:
            detector (CharucoDetector, optional): Charuco detector instance
            squares_x (int): Number of squares in X direction
            squares_y (int): Number of squares in Y direction
            square_length (float): Length of square side in meters
            marker_length (float): Length of marker side in meters
            dictionary_id (int): ArUco dictionary ID
            baseline (float): Baseline distance between cameras in meters (default: 0.1m = 100mm)
        """
        if detector is None:
            self.detector = CharucoDetector(
                squares_x, squares_y, square_length, marker_length, dictionary_id
            )
        else:
            self.detector = detector

        self.board = self.detector.get_board()

        # Store the baseline (distance between cameras)
        self.baseline = baseline

        # Create calibrators for left and right cameras
        self.left_calibrator = CameraCalibrator(self.detector)
        self.right_calibrator = CameraCalibrator(self.detector)

        # Lists to store stereo calibration data
        self.stereo_image_points = []
        self.stereo_object_points = []

    def add_stereo_pair(self, left_image, right_image, draw=False):
        """
        Process a stereo image pair for calibration.

        Args:
            left_image (numpy.ndarray): Left camera image
            right_image (numpy.ndarray): Right camera image
            draw (bool): Whether to draw detected markers and corners on the images

        Returns:
            tuple: (success, left_image_with_detections, right_image_with_detections)
                success: Whether the stereo pair was successfully processed
                left_image_with_detections: Left image with drawn detections (if draw=True)
                right_image_with_detections: Right image with drawn detections (if draw=True)
        """
        # Detect Charuco board in both images
        left_corners, left_ids, left_image_with_detections = self.detector.detect_board(left_image, draw)
        right_corners, right_ids, right_image_with_detections = self.detector.detect_board(right_image, draw)

        # If detection failed in either image, return failure
        if (left_corners is None or left_ids is None or len(left_corners) < 4 or
            right_corners is None or right_ids is None or len(right_ids) < 4):
            return False, left_image_with_detections, right_image_with_detections

        # Find common corners
        common_ids = np.intersect1d(left_ids, right_ids)

        if len(common_ids) < 4:
            return False, left_image_with_detections, right_image_with_detections

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
            row = id // self.detector.squares_x
            col = id % self.detector.squares_x
            object_point = np.array([col * self.detector.square_length, row * self.detector.square_length, 0], dtype=np.float32)

            # Add points to lists
            object_points.append(object_point)
            left_points.append(left_point)
            right_points.append(right_point)

        # Convert lists to numpy arrays
        object_points = np.array(object_points, dtype=np.float32)
        left_points = np.array(left_points, dtype=np.float32)
        right_points = np.array(right_points, dtype=np.float32)

        # Add points to stereo calibration data
        self.stereo_object_points.append(object_points)
        self.stereo_image_points.append((left_points, right_points))

        # Add images to individual camera calibrators
        self.left_calibrator.add_image(left_image, draw)
        self.right_calibrator.add_image(right_image, draw)

        return True, left_image_with_detections, right_image_with_detections

    def calibrate(self):
        """
        Perform stereo camera calibration.

        Returns:
            tuple: (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
                   R, T, E, F, R1, R2, P1, P2, Q)
                ret: Calibration error
                left_camera_matrix: Left camera matrix
                left_dist_coeffs: Left camera distortion coefficients
                right_camera_matrix: Right camera matrix
                right_dist_coeffs: Right camera distortion coefficients
                R: Rotation matrix between cameras
                T: Translation vector between cameras
                E: Essential matrix
                F: Fundamental matrix
                R1: Rectification rotation for left camera
                R2: Rectification rotation for right camera
                P1: Projection matrix for left camera
                P2: Projection matrix for right camera
                Q: Disparity-to-depth mapping matrix
        """
        if not self.stereo_object_points or not self.stereo_image_points:
            raise ValueError("No stereo calibration data available")

        # Calibrate individual cameras
        _, left_camera_matrix, left_dist_coeffs, _, _ = self.left_calibrator.calibrate()
        _, right_camera_matrix, right_dist_coeffs, _, _ = self.right_calibrator.calibrate()

        # Get image size
        image_size = self.left_calibrator.image_size

        # Prepare object points and image points for stereo calibration
        object_points = self.stereo_object_points
        left_points = [points[0] for points in self.stereo_image_points]
        right_points = [points[1] for points in self.stereo_image_points]

        # Perform stereo calibration with fixed baseline
        # First, do standard calibration
        ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = (
            cv2.stereoCalibrate(
                object_points, left_points, right_points,
                left_camera_matrix, left_dist_coeffs,
                right_camera_matrix, right_dist_coeffs,
                image_size, None, None,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
        )

        # Now, adjust the translation vector to match the known baseline
        # The baseline is the X component of the translation vector
        # We need to scale the entire translation vector to match our known baseline
        scale_factor = self.baseline / abs(T[0][0])
        T = T * scale_factor

        print(f"Adjusted translation vector to match baseline of {self.baseline*1000:.1f}mm")
        print(f"Original baseline from calibration: {abs(T[0][0]/scale_factor)*1000:.1f}mm")

        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_camera_matrix, left_dist_coeffs,
            right_camera_matrix, right_dist_coeffs,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9
        )

        return (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
                R, T, E, F, R1, R2, P1, P2, Q)

    def process_directories(self, left_dir, right_dir, draw=False, output_dir=None):
        """
        Process all image pairs in the specified directories for calibration.

        Args:
            left_dir (str): Directory containing left camera images
            right_dir (str): Directory containing right camera images
            draw (bool): Whether to draw detected markers and corners on the images
            output_dir (str, optional): Directory to save images with detections

        Returns:
            int: Number of successfully processed image pairs
        """
        # Create output directories if needed
        if draw and output_dir:
            os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)

        # Get all image files from left directory
        left_files = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Get all image files from right directory
        right_files = sorted([f for f in os.listdir(right_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Find common filenames
        common_files = set(left_files).intersection(set(right_files))

        # Process each image pair
        successful_pairs = 0
        for filename in common_files:
            # Load images
            left_path = os.path.join(left_dir, filename)
            right_path = os.path.join(right_dir, filename)

            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)

            if left_image is None or right_image is None:
                print(f"Failed to load image pair: {left_path}, {right_path}")
                continue

            # Process image pair
            success, left_with_detections, right_with_detections = self.add_stereo_pair(
                left_image, right_image, draw
            )

            if success:
                successful_pairs += 1
                print(f"Successfully processed pair: {filename}")

                # Save images with detections if requested
                if draw and output_dir:
                    if left_with_detections is not None:
                        cv2.imwrite(os.path.join(output_dir, 'left', filename), left_with_detections)
                    if right_with_detections is not None:
                        cv2.imwrite(os.path.join(output_dir, 'right', filename), right_with_detections)
            else:
                print(f"Failed to detect Charuco board in pair: {filename}")

        return successful_pairs
