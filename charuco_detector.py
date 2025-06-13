#!/usr/bin/env python3
"""
Utility functions for detecting Charuco boards in images.
"""

import cv2
import numpy as np

class CharucoDetector:
    def __init__(self, squares_x=9, squares_y=7, square_length=0.02, marker_length=0.015,
                dictionary_id=cv2.aruco.DICT_6X6_250):
        """
        Initialize the Charuco detector.

        Args:
            squares_x (int): Number of squares in X direction
            squares_y (int): Number of squares in Y direction
            square_length (float): Length of square side in meters
            marker_length (float): Length of marker side in meters
            dictionary_id (int): ArUco dictionary ID
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        # Create ArUco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        # Create Charuco board
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)

        # Create detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()

        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect_board(self, image, draw=False):
        """
        Detect Charuco board in the image.

        Args:
            image (numpy.ndarray): Input image
            draw (bool): Whether to draw detected markers and corners on the image

        Returns:
            tuple: (corners, ids, image_with_detections)
                corners: Detected Charuco corners
                ids: IDs of the detected corners
                image_with_detections: Image with drawn detections (if draw=True)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create a copy for drawing
        image_with_detections = image.copy() if draw else None

        # Detect ArUco markers
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)

        # If no markers detected, return empty results
        if marker_ids is None:
            return None, None, image_with_detections

        # Draw detected markers if requested
        if draw:
            cv2.aruco.drawDetectedMarkers(image_with_detections, marker_corners, marker_ids)

        # Interpolate Charuco corners
        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board)

        # If no Charuco corners detected, return empty results
        if response <= 0:
            return None, None, image_with_detections

        # Draw detected Charuco corners if requested
        if draw and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(image_with_detections, charuco_corners, charuco_ids)

        return charuco_corners, charuco_ids, image_with_detections

    def get_board(self):
        """
        Get the Charuco board object.

        Returns:
            cv2.aruco.CharucoBoard: The Charuco board
        """
        return self.board
