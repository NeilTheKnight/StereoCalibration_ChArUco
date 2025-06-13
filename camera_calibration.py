#!/usr/bin/env python3
"""
Camera calibration utilities using Charuco boards.
"""

import os
import cv2
import numpy as np
from charuco_detector import CharucoDetector

class CameraCalibrator:
    def __init__(self, detector=None, squares_x=9, squares_y=7, square_length=0.02, marker_length=0.015,
                dictionary_id=cv2.aruco.DICT_6X6_250):
        """
        Initialize the camera calibrator.

        Args:
            detector (CharucoDetector, optional): Charuco detector instance
            squares_x (int): Number of squares in X direction
            squares_y (int): Number of squares in Y direction
            square_length (float): Length of square side in meters
            marker_length (float): Length of marker side in meters
            dictionary_id (int): ArUco dictionary ID
        """
        if detector is None:
            self.detector = CharucoDetector(
                squares_x, squares_y, square_length, marker_length, dictionary_id
            )
        else:
            self.detector = detector

        self.board = self.detector.get_board()

        # Lists to store calibration data
        self.all_corners = []
        self.all_ids = []
        self.image_size = None

    def add_image(self, image, draw=False):
        """
        Process an image for calibration.

        Args:
            image (numpy.ndarray): Input image
            draw (bool): Whether to draw detected markers and corners on the image

        Returns:
            tuple: (success, image_with_detections)
                success: Whether the image was successfully processed
                image_with_detections: Image with drawn detections (if draw=True)
        """
        # Store image size
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        elif (image.shape[1], image.shape[0]) != self.image_size:
            raise ValueError("All images must have the same size")

        # Detect Charuco board
        corners, ids, image_with_detections = self.detector.detect_board(image, draw)

        # If detection failed, return failure
        if corners is None or ids is None or len(corners) < 4:
            return False, image_with_detections

        # Add detection to calibration data
        self.all_corners.append(corners)
        self.all_ids.append(ids)

        return True, image_with_detections

    def calibrate(self):
        """
        Perform camera calibration.

        Returns:
            tuple: (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
                ret: Calibration error
                camera_matrix: Camera matrix
                dist_coeffs: Distortion coefficients
                rvecs: Rotation vectors
                tvecs: Translation vectors
        """
        if not self.all_corners or not self.all_ids:
            raise ValueError("No calibration data available")

        # Prepare object points (3D points of the board corners)
        obj_points = []
        img_points = []

        # Get object points from the board
        for corners, ids in zip(self.all_corners, self.all_ids):
            # Create object points for the detected corners
            # For each corner ID, calculate its 3D position on the board
            board_obj_points = []
            for id in ids.flatten():
                # Calculate row and column based on ID
                row = id // self.detector.squares_x
                col = id % self.detector.squares_x
                # Create 3D point (X, Y, Z=0)
                point = np.array([col * self.detector.square_length,
                                 row * self.detector.square_length,
                                 0], dtype=np.float32)
                board_obj_points.append(point)

            # Convert to numpy array
            board_obj_points = np.array(board_obj_points, dtype=np.float32)

            # Add to object points and image points
            obj_points.append(board_obj_points)
            img_points.append(corners)

        # Initial camera matrix guess
        camera_matrix = np.array([
            [1000, 0, self.image_size[0] / 2],
            [0, 1000, self.image_size[1] / 2],
            [0, 0, 1]
        ])

        # Initial distortion coefficients
        dist_coeffs = np.zeros(5)

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, self.image_size, camera_matrix, dist_coeffs,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        return ret, camera_matrix, dist_coeffs, rvecs, tvecs

    def process_directory(self, directory, draw=False, output_dir=None):
        """
        Process all images in a directory for calibration.

        Args:
            directory (str): Directory containing images
            draw (bool): Whether to draw detected markers and corners on the images
            output_dir (str, optional): Directory to save images with detections

        Returns:
            int: Number of successfully processed images
        """
        # Create output directory if needed
        if draw and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Process each image
        successful_images = 0
        for image_file in image_files:
            # Load image
            image_path = os.path.join(directory, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Process image
            success, image_with_detections = self.add_image(image, draw)

            if success:
                successful_images += 1
                print(f"Successfully processed: {image_path}")

                # Save image with detections if requested
                if draw and output_dir and image_with_detections is not None:
                    output_path = os.path.join(output_dir, image_file)
                    cv2.imwrite(output_path, image_with_detections)
            else:
                print(f"Failed to detect Charuco board in: {image_path}")

        return successful_images
