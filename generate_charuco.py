#!/usr/bin/env python3
"""
Generate a Charuco board for camera calibration.
"""

import argparse
import cv2
import numpy as np

def generate_charuco_board(squares_x, squares_y, square_length, marker_length,
                          dictionary_id=cv2.aruco.DICT_6X6_250, output_file=None):
    """
    Generate a Charuco board with the specified parameters.

    Args:
        squares_x (int): Number of squares in X direction
        squares_y (int): Number of squares in Y direction
        square_length (float): Length of square side in meters
        marker_length (float): Length of marker side in meters
        dictionary_id (int): ArUco dictionary ID
        output_file (str): Path to save the generated board image

    Returns:
        numpy.ndarray: Generated Charuco board image
    """
    # Create ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # Create Charuco board
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

    # Generate board image
    board_image = board.generateImage((squares_x*200, squares_y*200))

    # Add a white border for better printing
    border_size = 50
    board_image_with_border = cv2.copyMakeBorder(
        board_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # Add text with board parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Squares: {squares_x}x{squares_y}, Square: {square_length}m, Marker: {marker_length}m"
    cv2.putText(board_image_with_border, text, (border_size, board_image_with_border.shape[0]-20),
                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the image if output file is specified
    if output_file:
        cv2.imwrite(output_file, board_image_with_border)
        print(f"Charuco board saved to {output_file}")

    return board_image_with_border

def main():
    parser = argparse.ArgumentParser(description='Generate a Charuco board for camera calibration')
    parser.add_argument('--squares_x', type=int, default=9, help='Number of squares in X direction')
    parser.add_argument('--squares_y', type=int, default=7, help='Number of squares in Y direction')
    parser.add_argument('--square_length', type=float, default=0.02, help='Length of square side in meters (20mm)')
    parser.add_argument('--marker_length', type=float, default=0.015, help='Length of marker side in meters (15mm)')
    parser.add_argument('--dictionary', type=str, default='DICT_6X6_250',
                        choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
                                'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
                                'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
                                'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000'],
                        help='ArUco dictionary to use')
    parser.add_argument('--output', type=str, default='charuco_board.png', help='Output image file')

    args = parser.parse_args()

    # Map string dictionary name to OpenCV constant
    dictionary_map = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    }

    dictionary_id = dictionary_map[args.dictionary]

    # Generate the board
    generate_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        dictionary_id,
        args.output
    )

if __name__ == "__main__":
    main()
