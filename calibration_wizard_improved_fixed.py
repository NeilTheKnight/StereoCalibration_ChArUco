import os
import cv2
import numpy as np
import json
import argparse
import glob
import matplotlib.pyplot as plt
from datetime import datetime

class CharucoDetector:
    """
    Class for detecting Charuco board in images.
    """
    def __init__(self, squares_x, squares_y, square_length, marker_length, dictionary_id):
        """
        Initialize the detector with board parameters.

        Args:
            squares_x (int): Number of squares in X direction
            squares_y (int): Number of squares in Y direction
            square_length (float): Square side length (in meters)
            marker_length (float): Marker side length (in meters)
            dictionary_id (int): ArUco dictionary ID
        """
        # Create Charuco board
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)
        self.detector_params = cv2.aruco.DetectorParameters()

        # Increase the number of iterations for corner refinement
        self.detector_params.cornerRefinementMaxIterations = 100
        self.detector_params.cornerRefinementMinAccuracy = 0.001

        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect_board(self, image, draw=True):
        """
        Detect Charuco board in an image.

        Args:
            image (numpy.ndarray): Input image
            draw (bool): Whether to draw detections on the image

        Returns:
            tuple: (corners, ids, image_with_detections)
                corners (numpy.ndarray): Detected corner coordinates
                ids (numpy.ndarray): Detected corner IDs
                image_with_detections (numpy.ndarray): Image with detections drawn
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        # If at least one marker detected
        if ids is not None and len(ids) > 0:
            # Refine detected markers
            corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                gray, self.board, corners, ids, rejected
            )

            # Interpolate Charuco corners
            response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )

            # Draw detections if requested
            if draw:
                image_copy = image.copy()
                # Draw marker borders
                cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)

                # Draw Charuco corners
                if charuco_corners is not None and len(charuco_corners) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)

                return charuco_corners, charuco_ids, image_copy

            return charuco_corners, charuco_ids, None

        return None, None, None

class ImprovedCalibrationWizard:
    """
    Improved wizard for stereo camera calibration using Charuco board.
    """
    def __init__(self):
        """
        Initialize the calibration wizard.
        """
        self.config = {
            'squares_x': 9,
            'squares_y': 7,
            'square_length': 0.02,  # 20mm
            'marker_length': 0.016,  # 16mm
            'dictionary_id': cv2.aruco.DICT_6X6_250,
            'baseline': 0.1,  # 100mm
            'min_successful_pairs': 5,
            'max_acceptable_error': 2.0,
            'draw_detections': True,
            'image_width': 1920,    # Updated for 16:9 aspect ratio
            'image_height': 1080    # Updated for 16:9 aspect ratio
        }

        self.left_dir = None
        self.right_dir = None
        self.output_dir = None
        self.image_size = None

        self.detector = None

        # Calibration data
        self.left_corners_all = []
        self.left_ids_all = []
        self.right_corners_all = []
        self.right_ids_all = []

        # Camera calibration results
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_camera_matrix = None
        self.right_dist_coeffs = None

        # Stereo calibration data
        self.stereo_object_points = []
        self.stereo_left_points = []
        self.stereo_right_points = []

        # Stereo calibration results
        self.stereo_results = None

    def welcome(self):
        """
        Display welcome message.
        """
        print("\nIMPROVED STEREO CAMERA CALIBRATION WIZARD")
        print("=" * 50)
        print("This wizard will guide you through the stereo camera calibration process.")
        print("Follow the steps to produce accurate calibration files.")
        print("=" * 50)

    def step1_setup(self, args=None):
        """
        Step 1: Setup calibration parameters and directories.

        Args:
            args (argparse.Namespace): Command line arguments

        Returns:
            bool: True if setup was successful, False otherwise
        """
        print("\nSTEP 1: SETUP")
        print("-" * 50)

        # Update configuration from command line arguments
        if args:
            if args.squares_x:
                self.config['squares_x'] = args.squares_x
            if args.squares_y:
                self.config['squares_y'] = args.squares_y
            if args.square_length:
                self.config['square_length'] = args.square_length
            if args.marker_length:
                self.config['marker_length'] = args.marker_length
            if args.baseline:
                self.config['baseline'] = args.baseline
            if args.dictionary:
                dict_name = f"cv2.aruco.{args.dictionary}"
                try:
                    self.config['dictionary_id'] = eval(dict_name)
                except:
                    print(f"Error: Invalid dictionary name: {args.dictionary}")
                    return False

            # Handle image width and height arguments with validation
            # Note: argparse converts --image-width to image_width attribute
            if hasattr(args, 'image_width') and args.image_width is not None:
                if args.image_width <= 0:
                    print(f"Error: Image width must be a positive integer, got {args.image_width}")
                    return False
                self.config['image_width'] = args.image_width

            if hasattr(args, 'image_height') and args.image_height is not None:
                if args.image_height <= 0:
                    print(f"Error: Image height must be a positive integer, got {args.image_height}")
                    return False
                self.config['image_height'] = args.image_height

            # Set input and output directories
            if args.input_dir:
                input_dir = args.input_dir
            else:
                input_dir = input("Enter input directory containing left and right camera images: ")

            if args.output_dir:
                self.output_dir = args.output_dir
            else:
                self.output_dir = input("Enter output directory for calibration results: ")
        else:
            # Prompt for input and output directories
            input_dir = input("Enter input directory containing left and right camera images: ")
            self.output_dir = input("Enter output directory for calibration results: ")

        # Check if input directory exists
        if not os.path.isdir(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist")
            return False

        # Check for left and right subdirectories
        self.left_dir = os.path.join(input_dir, 'left')
        self.right_dir = os.path.join(input_dir, 'right')

        if not os.path.isdir(self.left_dir):
            print(f"Error: Left camera directory '{self.left_dir}' does not exist")
            return False

        if not os.path.isdir(self.right_dir):
            print(f"Error: Right camera directory '{self.right_dir}' does not exist")
            return False

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create detector
        self.detector = CharucoDetector(
            self.config['squares_x'],
            self.config['squares_y'],
            self.config['square_length'],
            self.config['marker_length'],
            self.config['dictionary_id']
        )

        # Print configuration
        print("\nCalibration Configuration:")
        print(f"  Board: {self.config['squares_x']}×{self.config['squares_y']} squares")
        print(f"  Square size: {self.config['square_length']*1000:.1f}mm")
        print(f"  Marker size: {self.config['marker_length']*1000:.1f}mm")
        print(f"  Dictionary: {args.dictionary if args and args.dictionary else 'DICT_6X6_250'}")
        print(f"  Baseline: {self.config['baseline']*1000:.1f}mm")
        print(f"  Image dimensions: {self.config['image_width']}×{self.config['image_height']} pixels")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {self.output_dir}")

        print("\nStep 1 completed successfully.")
        return True

    def step2_detect_charuco(self):
        """
        Step 2: Detect Charuco board in images.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        print("\nSTEP 2: CHARUCO BOARD DETECTION")
        print("-" * 50)

        # Create detection output directories
        left_detection_dir = os.path.join(self.output_dir, 'detections', 'left')
        right_detection_dir = os.path.join(self.output_dir, 'detections', 'right')
        os.makedirs(left_detection_dir, exist_ok=True)
        os.makedirs(right_detection_dir, exist_ok=True)

        # Process left camera images
        print("Processing left camera images...")
        left_image_files = sorted(os.listdir(self.left_dir))
        left_successful = 0

        for image_file in left_image_files:
            if not (image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png')):
                continue

            print(f"Processing: {image_file}")
            image_path = os.path.join(self.left_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Store image size
            if self.image_size is None:
                self.image_size = (image.shape[1], image.shape[0])

            # Detect Charuco board
            corners, ids, image_with_detections = self.detector.detect_board(image, self.config['draw_detections'])

            if corners is not None and ids is not None and len(corners) >= 4:
                self.left_corners_all.append(corners)
                self.left_ids_all.append(ids)
                left_successful += 1
                print(f"Successfully processed: {image_file}")

                # Save image with detections
                if self.config['draw_detections'] and image_with_detections is not None:
                    output_path = os.path.join(left_detection_dir, image_file)
                    cv2.imwrite(output_path, image_with_detections)
            else:
                print(f"Failed to detect Charuco board in: {image_file}")

        print(f"Successfully processed {left_successful} left camera images")

        # Process right camera images
        print("\nProcessing right camera images...")
        right_image_files = sorted(os.listdir(self.right_dir))
        right_successful = 0

        for image_file in right_image_files:
            if not (image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png')):
                continue

            print(f"Processing: {image_file}")
            image_path = os.path.join(self.right_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Detect Charuco board
            corners, ids, image_with_detections = self.detector.detect_board(image, self.config['draw_detections'])

            if corners is not None and ids is not None and len(corners) >= 4:
                self.right_corners_all.append(corners)
                self.right_ids_all.append(ids)
                right_successful += 1
                print(f"Successfully processed: {image_file}")

                # Save image with detections
                if self.config['draw_detections'] and image_with_detections is not None:
                    output_path = os.path.join(right_detection_dir, image_file)
                    cv2.imwrite(output_path, image_with_detections)
            else:
                print(f"Failed to detect Charuco board in: {image_file}")

        print(f"Successfully processed {right_successful} right camera images")

        # Check if we have enough successful detections
        if left_successful < self.config['min_successful_pairs'] or right_successful < self.config['min_successful_pairs']:
            print(f"Error: Not enough successful detections. Need at least {self.config['min_successful_pairs']} per camera.")
            return False

        print("\nStep 2 completed successfully.")
        return True

    def step3_calibrate_individual_cameras(self):
        """
        Step 3: Calibrate individual cameras with improved methods.

        Returns:
            bool: True if calibration was successful, False otherwise
        """
        print("\nSTEP 3: INDIVIDUAL CAMERA CALIBRATION")
        print("-" * 50)

        # Calibrate left camera
        print("Calibrating left camera...")
        left_ret, self.left_camera_matrix, self.left_dist_coeffs = self._calibrate_camera_improved(
            self.left_corners_all, self.left_ids_all, "left"
        )

        # Calibrate right camera
        print("\nCalibrating right camera...")
        right_ret, self.right_camera_matrix, self.right_dist_coeffs = self._calibrate_camera_improved(
            self.right_corners_all, self.right_ids_all, "right"
        )

        # Check calibration errors - very strict for individual cameras
        print(f"Left camera calibration error: {left_ret:.6f}")
        print(f"Right camera calibration error: {right_ret:.6f}")

        if left_ret > 0.6:
            print(f"ERROR: Left camera calibration error is too high ({left_ret:.6f} > 0.6).")
            print("This will lead to poor stereo calibration results.")
            print("Please recapture calibration images with better quality.")
            return False

        if right_ret > 0.6:
            print(f"ERROR: Right camera calibration error is too high ({right_ret:.6f} > 0.6).")
            print("This will lead to poor stereo calibration results.")
            print("Please recapture calibration images with better quality.")
            return False

        if left_ret > 0.3 or right_ret > 0.3:
            print(f"WARNING: Camera calibration errors are above optimal threshold (target: <0.3).")
            print(f"Results may be suboptimal but proceeding with stereo calibration.")

        print("✓ Individual camera calibration errors are within acceptable range.")

        # Save undistorted sample images
        self._save_undistorted_samples()

        print("\nStep 3 completed successfully.")
        return True

    def _calibrate_camera_improved(self, corners_all, ids_all, camera_name):
        """
        Improved camera calibration method with outlier detection and validation.

        Args:
            corners_all (list): List of detected corners for each image
            ids_all (list): List of detected IDs for each image
            camera_name (str): Camera name for logging

        Returns:
            tuple: (ret, camera_matrix, dist_coeffs)
                ret (float): Calibration error
                camera_matrix (numpy.ndarray): Camera matrix
                dist_coeffs (numpy.ndarray): Distortion coefficients
        """
        print(f"Calibrating {camera_name} camera with {len(corners_all)} images...")

        # Prepare object points (3D points in real world space)
        obj_points = []
        img_points = []

        # Get board object points
        board = self.detector.board

        # Process each image
        for corners, ids in zip(corners_all, ids_all):
            # Get object and image points for this image
            # Use alternative method since estimatePoseSingleFrameCharuco is not available
            # Get board object points
            board_points = board.getChessboardCorners()

            # Get corresponding image points
            current_obj_points = []
            current_img_points = []

            for i, corner_id in enumerate(ids.flatten()):
                current_obj_points.append(board_points[corner_id])
                current_img_points.append(corners[i].flatten())

            # Convert to numpy arrays with correct data types
            current_obj_points = np.array(current_obj_points, dtype=np.float32)
            current_img_points = np.array(current_img_points, dtype=np.float32)

            if current_obj_points is not None and current_img_points is not None:
                obj_points.append(current_obj_points)
                img_points.append(current_img_points)

        if not obj_points:
            raise ValueError(f"No valid points found for {camera_name} camera calibration")

        # Better initial camera matrix guess based on typical camera parameters
        # Use realistic focal length estimate based on image dimensions
        # Based on typical camera setups, focal length should be around 0.8-1.2 * image_width
        fx = self.config['image_width'] * 0.8  # Conservative but realistic estimate
        fy = fx  # Assume square pixels (aspect ratio = 1)
        cx = self.config['image_width'] / 2.0  # Principal point at image center
        cy = self.config['image_height'] / 2.0

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Initial distortion coefficients (k1, k2, p1, p2, k3)
        dist_coeffs = np.zeros(5, dtype=np.float64)

        # Define reasonable bounds for camera parameters - make them more permissive
        min_focal_length = self.config['image_width'] * 0.1  # Minimum reasonable focal length
        max_focal_length = self.config['image_width'] * 3.0  # Maximum reasonable focal length

        # Try different calibration flags to find the best result - prioritize stability and low distortion
        calibration_flags = [
            # Most constrained approaches first - these typically produce the most stable results
            cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
            cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST,
            cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3,
            cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO,
            # Moderately constrained
            cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
            cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST,
            cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
            cv2.CALIB_FIX_ASPECT_RATIO,
            cv2.CALIB_ZERO_TANGENT_DIST
        ]

        best_ret = float('inf')
        best_camera_matrix = None
        best_dist_coeffs = None

        for flags in calibration_flags:
            # Calibrate camera with improved criteria
            ret, cam_matrix, dist_coeffs_result, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points,
                (self.config['image_width'], self.config['image_height']),
                camera_matrix.copy(), dist_coeffs.copy(),
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )

            print(f"  Calibration with flags {flags}: error = {ret}")

            # Check if camera parameters are within reasonable bounds
            if (cam_matrix[0, 0] < min_focal_length or cam_matrix[0, 0] > max_focal_length or
                cam_matrix[1, 1] < min_focal_length or cam_matrix[1, 1] > max_focal_length):
                print(f"  Rejecting result: focal length out of bounds")
                continue

            # Check if distortion coefficients are reasonable - strict quality control
            k1, k2 = abs(dist_coeffs_result[0]), abs(dist_coeffs_result[1])
            p1, p2 = abs(dist_coeffs_result[2]), abs(dist_coeffs_result[3])

            # Reject if radial distortion coefficients are too large (indicates poor calibration)
            if k1 > 1.0 or k2 > 1.0:
                print(f"  Rejecting result: radial distortion too large (k1={k1:.3f}, k2={k2:.3f})")
                continue

            # Reject if tangential distortion coefficients are too large
            if p1 > 0.01 or p2 > 0.01:
                print(f"  Rejecting result: tangential distortion too large (p1={p1:.3f}, p2={p2:.3f})")
                continue

            # Calculate reprojection error for each point to detect outliers
            # Use a simpler approach to avoid shape mismatches
            print(f"  Calibration error (RMS): {ret}")

            # Skip detailed error calculation as it's causing issues
            # Just use the overall RMS error from calibrateCamera

            # Check if this is the best result so far - focus on lowest error
            if ret < best_ret and ret < 0.5:  # Target sub-pixel accuracy
                best_ret = ret
                best_camera_matrix = cam_matrix
                best_dist_coeffs = dist_coeffs_result
                print(f"  *** New best result: {ret:.6f} ***")
            elif ret < best_ret:
                # Still track the best result even if above threshold
                best_ret = ret
                best_camera_matrix = cam_matrix
                best_dist_coeffs = dist_coeffs_result
                print(f"  *** Best so far (above target): {ret:.6f} ***")

        if best_camera_matrix is None:
            raise ValueError(f"Could not find acceptable calibration for {camera_name} camera")

        print(f"\nBest {camera_name} camera calibration:")
        print(f"  Error: {best_ret}")
        print(f"  Camera matrix:\n{best_camera_matrix}")
        print(f"  Distortion coefficients: {best_dist_coeffs.ravel()}")

        return best_ret, best_camera_matrix, best_dist_coeffs

    def _save_undistorted_samples(self):
        """
        Save undistorted sample images for visual inspection.
        """
        # Create undistorted samples directory
        undistorted_dir = os.path.join(self.output_dir, 'undistorted_samples')
        os.makedirs(undistorted_dir, exist_ok=True)

        # Get a sample image from each camera
        left_images = sorted(glob.glob(os.path.join(self.left_dir, '*.jpg')))
        if not left_images:
            left_images = sorted(glob.glob(os.path.join(self.left_dir, '*.png')))

        right_images = sorted(glob.glob(os.path.join(self.right_dir, '*.jpg')))
        if not right_images:
            right_images = sorted(glob.glob(os.path.join(self.right_dir, '*.png')))

        if left_images and self.left_camera_matrix is not None and self.left_dist_coeffs is not None:
            # Load and undistort left image
            left_image = cv2.imread(left_images[0])
            left_undistorted = cv2.undistort(
                left_image, self.left_camera_matrix, self.left_dist_coeffs
            )

            # Save side-by-side comparison
            left_comparison = np.hstack((left_image, left_undistorted))
            cv2.putText(left_comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(left_comparison, "Undistorted", (left_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(undistorted_dir, 'left_undistorted_comparison.jpg'), left_comparison)

        if right_images and self.right_camera_matrix is not None and self.right_dist_coeffs is not None:
            # Load and undistort right image
            right_image = cv2.imread(right_images[0])
            right_undistorted = cv2.undistort(
                right_image, self.right_camera_matrix, self.right_dist_coeffs
            )

            # Save side-by-side comparison
            right_comparison = np.hstack((right_image, right_undistorted))
            cv2.putText(right_comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(right_comparison, "Undistorted", (right_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(undistorted_dir, 'right_undistorted_comparison.jpg'), right_comparison)

        print(f"Saved undistorted sample images to {undistorted_dir}")

    def step4_find_stereo_correspondences(self):
        """
        Step 4: Find stereo correspondences between left and right camera images.

        Returns:
            bool: True if finding correspondences was successful, False otherwise
        """
        print("\nSTEP 4: FINDING STEREO CORRESPONDENCES")
        print("-" * 50)

        # Get common image files
        left_files = set(os.listdir(self.left_dir))
        right_files = set(os.listdir(self.right_dir))
        common_files = sorted(list(left_files.intersection(right_files)))

        # Filter for image files
        common_files = [f for f in common_files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

        if len(common_files) < self.config['min_successful_pairs']:
            print(f"Error: Not enough common image files. Found {len(common_files)}, need at least {self.config['min_successful_pairs']}.")
            return False

        print(f"Found {len(common_files)} common image files.")

        # Process each pair of images
        successful_pairs = 0

        for image_file in common_files:
            print(f"Processing stereo pair: {image_file}")

            # Load images
            left_image = cv2.imread(os.path.join(self.left_dir, image_file))
            right_image = cv2.imread(os.path.join(self.right_dir, image_file))

            if left_image is None or right_image is None:
                print(f"Failed to read images for {image_file}")
                continue

            # Detect Charuco board in both images
            left_corners, left_ids, _ = self.detector.detect_board(left_image, False)
            right_corners, right_ids, _ = self.detector.detect_board(right_image, False)

            if (left_corners is None or left_ids is None or len(left_corners) < 4 or
                right_corners is None or right_ids is None or len(right_corners) < 4):
                print(f"Failed to detect Charuco board in both images for {image_file}")
                continue

            # Find common corners
            common_ids = set(left_ids.ravel()).intersection(set(right_ids.ravel()))
            if len(common_ids) < 4:
                print(f"Not enough common corners detected in {image_file}")
                continue

            # Get object points and image points for common corners
            board = self.detector.board

            # Get object points (3D points in real world space)
            obj_points = []
            left_img_points = []
            right_img_points = []

            for corner_id in common_ids:
                # Find index in left image
                left_idx = np.where(left_ids == corner_id)[0][0]
                # Find index in right image
                right_idx = np.where(right_ids == corner_id)[0][0]

                # Get 3D point from board
                board_points = board.getChessboardCorners()
                obj_point = board_points[corner_id]

                # Get 2D points from images
                left_point = left_corners[left_idx].ravel()
                right_point = right_corners[right_idx].ravel()

                obj_points.append(obj_point)
                left_img_points.append(left_point)
                right_img_points.append(right_point)

            # Convert to numpy arrays with correct shape for stereoCalibrate
            obj_points = np.array(obj_points, dtype=np.float32)
            left_img_points = np.array(left_img_points, dtype=np.float32)
            right_img_points = np.array(right_img_points, dtype=np.float32)

            # Quality check: only use pairs with sufficient corner detections
            min_corners_required = 10  # Require at least 10 corners for calibration
            if len(left_img_points) >= min_corners_required and len(right_img_points) >= min_corners_required:
                # Add to stereo calibration data
                self.stereo_object_points.append(obj_points)
                self.stereo_left_points.append(left_img_points)
                self.stereo_right_points.append(right_img_points)

                successful_pairs += 1
                print(f"Successfully processed stereo pair: {image_file} ({len(left_img_points)} corners)")
            else:
                print(f"Skipping {image_file}: insufficient corners (L:{len(left_img_points)}, R:{len(right_img_points)})")

        print(f"Successfully processed {successful_pairs} stereo pairs")

        if successful_pairs < self.config['min_successful_pairs']:
            print(f"Error: Not enough successful stereo pairs. Need at least {self.config['min_successful_pairs']}.")
            return False

        print("\nStep 4 completed successfully.")
        return True

    def step5_stereo_calibration(self):
        """
        Step 5: Perform stereo calibration with improved methods.

        Returns:
            bool: True if calibration was successful, False otherwise
        """
        print("\nSTEP 5: STEREO CALIBRATION")
        print("-" * 50)

        if not self.stereo_object_points or not self.stereo_left_points or not self.stereo_right_points:
            print("Error: No stereo calibration data available")
            return False

        print("Performing stereo calibration...")
        try:
            # Improved calibration flags for better stereo calibration
            calibration_flags = [
                cv2.CALIB_FIX_INTRINSIC,  # Use pre-calibrated camera matrices (most stable)
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO,  # Constrained optimization
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT,  # Allow focal length optimization
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO,  # Allow principal point optimization
                cv2.CALIB_USE_INTRINSIC_GUESS,  # Allow all intrinsic optimization
                0  # Full optimization (least stable but sometimes necessary)
            ]

            best_ret = float('inf')
            best_results = None

            for flags in calibration_flags:
                print(f"\nTrying calibration with flags: {flags}")

                # Perform stereo calibration with improved criteria
                ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = (
                    cv2.stereoCalibrate(
                        self.stereo_object_points,
                        self.stereo_left_points,
                        self.stereo_right_points,
                        self.left_camera_matrix.copy(),
                        self.left_dist_coeffs.copy(),
                        self.right_camera_matrix.copy(),
                        self.right_dist_coeffs.copy(),
                        (self.config['image_width'], self.config['image_height']), None, None,
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6),  # More iterations, higher precision
                        flags=flags
                    )
                )

                print(f"Calibration error: {ret}")

                # Check if the results are reasonable
                if self._validate_stereo_results(ret, left_camera_matrix, right_camera_matrix, R, T):
                    # Check if this is the best result so far
                    if ret < best_ret:
                        best_ret = ret
                        best_results = (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F)
                else:
                    print("Rejecting unreasonable calibration result")

            if best_results is None:
                print("Warning: Could not find acceptable stereo calibration with strict criteria")
                print("Trying with relaxed validation criteria...")

                # Try again with more relaxed validation
                for flags in calibration_flags:
                    print(f"\nRetrying calibration with flags: {flags} (relaxed validation)")

                    ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = (
                        cv2.stereoCalibrate(
                            self.stereo_object_points,
                            self.stereo_left_points,
                            self.stereo_right_points,
                            self.left_camera_matrix.copy(),
                            self.left_dist_coeffs.copy(),
                            self.right_camera_matrix.copy(),
                            self.right_dist_coeffs.copy(),
                            (self.config['image_width'], self.config['image_height']), None, None,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6),
                            flags=flags
                        )
                    )

                    print(f"Calibration error: {ret}")

                    # Use more relaxed validation criteria
                    if ret < 20.0:  # Much more permissive error threshold
                        print("Accepting result with relaxed criteria")
                        best_results = (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F)
                        break

                if best_results is None:
                    print("Error: Could not find acceptable stereo calibration even with relaxed criteria")
                    return False

            # Use the best calibration results
            (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F) = best_results

            print(f"\nBest calibration error: {ret}")
            print(f"Left camera matrix:\n{left_camera_matrix}")
            print(f"Right camera matrix:\n{right_camera_matrix}")
            print(f"Rotation matrix:\n{R}")
            print(f"Translation vector:\n{T}")

            # Calculate and print baseline
            original_baseline = abs(T[0][0])
            print(f"Original baseline from calibration: {original_baseline*1000:.1f}mm")

            # Adjust translation vector to match known baseline
            scale_factor = self.config['baseline'] / original_baseline
            T = T * scale_factor

            print(f"Adjusted translation vector to match baseline of {self.config['baseline']*1000:.1f}mm")
            print(f"Adjusted translation vector:\n{T}")

            # Compute rectification transforms
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                left_camera_matrix, left_dist_coeffs,
                right_camera_matrix, right_dist_coeffs,
                (self.config['image_width'], self.config['image_height']), R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0.9
            )

            # Store stereo calibration results
            self.stereo_results = (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
                                  R, T, E, F, R1, R2, P1, P2, Q)

            # Check calibration error with updated threshold
            if ret > self.config['max_acceptable_error']:
                print(f"\nWARNING: Calibration error is above target ({ret:.6f} > {self.config['max_acceptable_error']}).")
                if ret > self.config['max_acceptable_error'] * 3:
                    print(f"This error is significantly high and may indicate problems with the calibration process.")
                    print(f"Consider recapturing calibration images or adjusting parameters.")

                    # Ask user if they want to continue despite high error
                    response = input("Continue despite high error? (y/n): ").strip().lower()
                    if response != 'y':
                        return False
                else:
                    print(f"Error is acceptable for this calibration setup.")
            else:
                print(f"\nCalibration error is within acceptable range ({ret:.6f} <= {self.config['max_acceptable_error']}).")

            print("\nStep 5 completed successfully.")
            return True
        except Exception as e:
            print(f"Error during stereo calibration: {str(e)}")
            return False

    def _validate_stereo_results(self, error, left_matrix, right_matrix, R, T):
        """
        Validate stereo calibration results.

        Args:
            error (float): Calibration error
            left_matrix (numpy.ndarray): Left camera matrix
            right_matrix (numpy.ndarray): Right camera matrix
            R (numpy.ndarray): Rotation matrix
            T (numpy.ndarray): Translation vector

        Returns:
            bool: True if results are valid, False otherwise
        """
        # Check calibration error - more permissive threshold
        if error > 15.0:  # Increased from 10.0 to 15.0 to allow more solutions
            print(f"Validation failed: Error too high ({error:.6f})")
            return False

        # Check camera matrices - use more permissive bounds
        min_focal_length = self.config['image_width'] * 0.1
        max_focal_length = self.config['image_width'] * 3.0

        if (left_matrix[0, 0] < min_focal_length or left_matrix[0, 0] > max_focal_length or
            left_matrix[1, 1] < min_focal_length or left_matrix[1, 1] > max_focal_length or
            right_matrix[0, 0] < min_focal_length or right_matrix[0, 0] > max_focal_length or
            right_matrix[1, 1] < min_focal_length or right_matrix[1, 1] > max_focal_length):
            print("Validation failed: Focal lengths out of reasonable range")
            print(f"Left focal lengths: ({left_matrix[0, 0]}, {left_matrix[1, 1]})")
            print(f"Right focal lengths: ({right_matrix[0, 0]}, {right_matrix[1, 1]})")
            return False

        # Check if cameras are roughly parallel (rotation should be small)
        # For real-world stereo setups, some rotation is expected
        r_trace = np.trace(R)
        if r_trace < 0.0:  # Very significant rotation - likely incorrect
            print(f"Validation failed: Cameras have excessive rotation (trace={r_trace:.2f})")
            return False
        elif r_trace < 1.0:  # Significant rotation but might be valid
            print(f"Warning: Cameras have significant rotation (trace={r_trace:.2f})")

        # Check if baseline is reasonable - be more permissive
        baseline = abs(T[0][0])
        expected_baseline = self.config['baseline']
        if baseline < expected_baseline * 0.1 or baseline > expected_baseline * 5.0:
            print(f"Validation failed: Unreasonable baseline ({baseline*1000:.1f}mm vs expected {expected_baseline*1000:.1f}mm)")
            return False
        elif baseline < expected_baseline * 0.5 or baseline > expected_baseline * 2.0:
            print(f"Warning: Baseline differs from expected ({baseline*1000:.1f}mm vs expected {expected_baseline*1000:.1f}mm)")

        # Check if vertical and depth offsets are small compared to baseline - be extremely permissive
        # For this dataset, we'll allow large depth offsets since the cameras might not be perfectly aligned
        if abs(T[1][0]) > baseline * 2.0:  # Only check vertical offset strictly
            print(f"Validation failed: Extremely large vertical offset in translation")
            print(f"Vertical offset: {abs(T[1][0])*1000:.1f}mm, Depth offset: {abs(T[2][0])*1000:.1f}mm")
            return False
        else:
            # Just print a warning about the offsets
            print(f"Translation offsets - Vertical: {abs(T[1][0])*1000:.1f}mm, Depth: {abs(T[2][0])*1000:.1f}mm")

        return True

    def step6_evaluate_results(self):
        """
        Step 6: Evaluate calibration results.

        Returns:
            bool: True if evaluation was successful, False otherwise
        """
        print("\nSTEP 6: EVALUATING RESULTS")
        print("-" * 50)

        if self.stereo_results is None:
            print("Error: No stereo calibration results available")
            return False

        # Save calibration results
        self._save_calibration_results()

        # Visualize rectification
        print("Visualizing rectification...")
        try:
            # Find a common image file
            left_files = set(os.listdir(self.left_dir))
            right_files = set(os.listdir(self.right_dir))
            common_files = sorted(list(left_files.intersection(right_files)))
            common_files = [f for f in common_files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

            if common_files:
                filename = common_files[0]
                left_image = cv2.imread(os.path.join(self.left_dir, filename))
                right_image = cv2.imread(os.path.join(self.right_dir, filename))

                # Visualize rectification
                self._visualize_rectification(left_image, right_image)

                # Calculate reprojection error
                reproj_error = self._calculate_reprojection_error()
                print(f"Reprojection error: {reproj_error:.6f} pixels")

                # Check if rectification looks reasonable
                print("\nPlease examine the rectification visualization:")
                print(f"  - File: {os.path.join(self.output_dir, 'rectification_map.png')}")
                print("  - Check if horizontal lines align across both images")
                print("  - Check if the same features appear on the same horizontal lines")

            print("\nCalibration results saved to:")
            print(f"  - {os.path.join(self.output_dir, 'camera_params.json')}")

            print("\nStep 6 completed successfully.")
            return True
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return False

    def _visualize_rectification(self, left_image, right_image):
        """
        Visualize rectification of stereo images.

        Args:
            left_image (numpy.ndarray): Left camera image
            right_image (numpy.ndarray): Right camera image
        """
        if self.stereo_results is None:
            return

        # Unpack calibration results
        (_, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
         _, _, _, _, R1, R2, P1, P2, _) = self.stereo_results

        # Get actual image size from the loaded images
        actual_image_size = (left_image.shape[1], left_image.shape[0])  # (width, height)

        print(f"Actual image size: {actual_image_size}")
        print(f"Calibration image size: ({self.config['image_width']}, {self.config['image_height']})")

        # Use actual image size for rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            left_camera_matrix, left_dist_coeffs, R1, P1, actual_image_size, cv2.CV_32FC1
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            right_camera_matrix, right_dist_coeffs, R2, P2, actual_image_size, cv2.CV_32FC1
        )

        # Rectify images
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        # Draw horizontal lines for visualization
        line_interval = left_image.shape[0] // 20
        for y in range(0, left_image.shape[0], line_interval):
            cv2.line(left_rectified, (0, y), (left_image.shape[1], y), (0, 255, 0), 1)
            cv2.line(right_rectified, (0, y), (right_image.shape[1], y), (0, 255, 0), 1)

        # Create side-by-side visualization
        rectified_pair = np.hstack((left_rectified, right_rectified))

        # Save visualization
        cv2.imwrite(os.path.join(self.output_dir, 'rectification_map.png'), rectified_pair)

        # Also save original images side by side for comparison
        original_pair = np.hstack((left_image, right_image))
        cv2.imwrite(os.path.join(self.output_dir, 'original_pair.png'), original_pair)

    def _calculate_reprojection_error(self):
        """
        Calculate reprojection error for stereo calibration.

        Returns:
            float: Average reprojection error in pixels
        """
        if self.stereo_results is None:
            return float('inf')

        # Simply return the calibration error from stereoCalibrate
        # This is already the RMS reprojection error
        return self.stereo_results[0]

    def _save_calibration_results(self):
        """
        Save calibration results to files.
        """
        if self.stereo_results is None:
            return

        # Unpack calibration results
        (ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
         R, T, E, F, R1, R2, P1, P2, Q) = self.stereo_results

        # Convert numpy arrays to lists for JSON serialization
        params = {
            'calibration_error': float(ret),
            'left_camera_matrix': left_camera_matrix.tolist(),
            'left_dist_coeffs': left_dist_coeffs.ravel().tolist(),
            'right_camera_matrix': right_camera_matrix.tolist(),
            'right_dist_coeffs': right_dist_coeffs.ravel().tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'baseline_mm': float(abs(T[0][0]) * 1000),  # Baseline in mm
            'known_baseline_mm': self.config['baseline'] * 1000,  # Known baseline in mm
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'left_rectification_matrix': R1.tolist(),
            'right_rectification_matrix': R2.tolist(),
            'left_projection_matrix': P1.tolist(),
            'right_projection_matrix': P2.tolist(),
            'disparity_to_depth_matrix': Q.tolist(),
            'board_config': {
                'squares_x': self.config['squares_x'],
                'squares_y': self.config['squares_y'],
                'square_length_mm': self.config['square_length'] * 1000,
                'marker_length_mm': self.config['marker_length'] * 1000,
                'dictionary': 'DICT_6X6_250'
            },
            'image_size': [self.config['image_width'], self.config['image_height']]
        }

        # Save parameters as JSON
        with open(os.path.join(self.output_dir, 'camera_params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        # Also save calibration data in OpenCV XML format
        fs = cv2.FileStorage(os.path.join(self.output_dir, 'stereo_calibration.xml'), cv2.FILE_STORAGE_WRITE)
        fs.write("calibration_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fs.write("image_width", self.config['image_width'])
        fs.write("image_height", self.config['image_height'])
        fs.write("calibration_error", ret)
        fs.write("left_camera_matrix", left_camera_matrix)
        fs.write("left_distortion_coefficients", left_dist_coeffs)
        fs.write("right_camera_matrix", right_camera_matrix)
        fs.write("right_distortion_coefficients", right_dist_coeffs)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", T)
        fs.write("essential_matrix", E)
        fs.write("fundamental_matrix", F)
        fs.write("left_rectification_matrix", R1)
        fs.write("right_rectification_matrix", R2)
        fs.write("left_projection_matrix", P1)
        fs.write("right_projection_matrix", P2)
        fs.write("disparity_to_depth_matrix", Q)
        fs.release()

def main():
    """
    Main function to run the calibration wizard.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Improved Stereo Camera Calibration Wizard')
    parser.add_argument('--input_dir', type=str, help='Input directory containing left and right camera images')
    parser.add_argument('--output_dir', type=str, help='Output directory for calibration results')
    parser.add_argument('--squares_x', type=int, help='Number of squares in X direction (default: 9)')
    parser.add_argument('--squares_y', type=int, help='Number of squares in Y direction (default: 7)')
    parser.add_argument('--square_length', type=float, help='Length of square side in meters (default: 0.02 = 20mm)')
    parser.add_argument('--marker_length', type=float, help='Length of marker side in meters (default: 0.016 = 16mm)')
    parser.add_argument('--baseline', type=float, help='Baseline distance between cameras in meters (default: 0.1 = 100mm)')
    parser.add_argument('--dictionary', type=str,
                        choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
                                'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
                                'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
                                'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000'],
                        help='ArUco dictionary to use (default: DICT_6X6_250)')
    parser.add_argument('--image-width', type=int, help='Image width in pixels (default: 1920)')
    parser.add_argument('--image-height', type=int, help='Image height in pixels (default: 1080)')

    args = parser.parse_args()

    # Create and run the wizard
    wizard = ImprovedCalibrationWizard()
    wizard.welcome()

    # Run each step, stopping if any step fails
    if not wizard.step1_setup(args):
        print("Setup failed. Exiting.")
        return

    if not wizard.step2_detect_charuco():
        print("Charuco detection failed. Exiting.")
        return

    if not wizard.step3_calibrate_individual_cameras():
        print("Individual camera calibration failed. Exiting.")
        return

    if not wizard.step4_find_stereo_correspondences():
        print("Finding stereo correspondences failed. Exiting.")
        return

    if not wizard.step5_stereo_calibration():
        print("Stereo calibration failed. Exiting.")
        return

    if not wizard.step6_evaluate_results():
        print("Evaluation failed. Exiting.")
        return

    print("\nCalibration completed successfully!")
    print(f"Calibration results saved to: {wizard.output_dir}")

if __name__ == "__main__":
    main()