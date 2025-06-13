# Stereo Camera Calibration Guide

This document provides a comprehensive guide for performing stereo camera calibration using the improved calibration wizard.

## Overview

Stereo camera calibration is essential for applications requiring depth perception, such as 3D reconstruction, object detection, and pupillary distance measurement. The calibration process determines the intrinsic parameters of each camera (focal length, principal point, distortion coefficients) and the extrinsic parameters (rotation and translation) between the two cameras.

## Required Equipment

1. **Stereo Camera Setup**:
   - Two identical cameras mounted with a fixed baseline of 100mm
   - Cameras should be roughly parallel and at the same height
   - Resolution: 1920×1080 pixels

2. **Charuco Board**:
   - 9×7 internal corners (squares)
   - Square size: 20mm
   - Marker size: 16mm
   - Dictionary: DICT_6X6_250
   - You can generate a Charuco board using the `generate_charuco.py` script

## Scripts and Programs

The following scripts are used in the calibration process:

0. **Run Setup**
   - `setup.bat` - The one-click setup script for Windows. Just run it. 
   - `setup.sh` - The one-click setup script for Linux. 

1. **Main Calibration Scripts**:
   - `calibration_wizard_improved_fixed.py` - The improved calibration wizard that guides you through the calibration process
   - `run_improved_calibration_fixed.bat` - Batch file to run the improved calibration wizard

2. **Utility Scripts**:
   - `camera_calibration.py` - Contains camera calibration functions
   - `charuco_detector.py` - Contains functions for detecting Charuco boards
   - `generate_charuco.py` - Script to generate Charuco boards
   - `visualize_rectification.py` - Script to visualize rectification results
   - `debug_charuco_detection.py` - Script to debug Charuco board detection
   - `debug_stereo_matching.py` - Script to debug stereo matching
   - `simple_visualize.py` - Simple visualization script
   - `stereo_calibration.py` - Contains stereo calibration functions

3. **Setup Script**:
   - `setup.bat` - Script to set up the Python environment

## Directory Structure

```
calibration_charuco/
├── input_images/
│   ├── left/          # Left camera images
│   └── right/         # Right camera images
├── calibration_results_test/  # Output directory for calibration results
├── old/
│   └── historyProg/   # Directory for old/unused scripts
├── calibration_wizard_improved_fixed.py
├── run_improved_calibration_fixed.bat
└── [utility scripts]
```

## Step-by-Step Calibration Procedure

### 1. Prepare the Environment

1. Ensure you have Python installed with OpenCV.
2. Run `setup.bat` to set up the Python environment if not already done.

### 2. Capture Calibration Images

1. Mount the stereo cameras with a baseline of approximately 100mm.
2. Place the Charuco board at various positions and orientations in front of the cameras.
3. Ensure the board is fully visible in both cameras for each capture.
4. Capture at least 9-15 pairs of images.
5. Save the left camera images in the `input_images/left/` directory.
6. Save the right camera images in the `input_images/right/` directory.
7. Ensure corresponding images from left and right cameras have the same filename.

### 3. Run the Calibration Wizard

1. Open a command prompt in the calibration directory.
2. Run the calibration wizard using the batch file:
   ```
   .\run_improved_calibration_fixed.bat
   ```
   
   Alternatively, you can run the Python script directly with custom parameters:
   ```
   python calibration_wizard_improved_fixed.py --input_dir input_images --output_dir calibration_results_test --squares_x 9 --squares_y 7 --square_length 0.02 --marker_length 0.016 --baseline 0.1 --dictionary DICT_6X6_250
   ```

### 4. Follow the Wizard Steps

The calibration wizard will guide you through the following steps:

1. **Setup**: Configure calibration parameters and directories.
2. **Charuco Board Detection**: Detect the Charuco board in all images.
3. **Individual Camera Calibration**: Calibrate each camera separately.
4. **Find Stereo Correspondences**: Find corresponding points between left and right images.
5. **Stereo Calibration**: Perform stereo calibration to determine the relationship between cameras.
6. **Evaluate Results**: Evaluate the calibration results and save the calibration files.

### 5. Examine the Results

After calibration, examine the following:

1. **Calibration Error**: The reprojection error should be less than 2.0 pixels. Lower is better.
2. **Rectification Visualization**: Check if horizontal lines align across both images.
3. **Camera Parameters**: Verify that the camera matrices and distortion coefficients are reasonable.
4. **Baseline**: Confirm that the calculated baseline matches the expected value (100mm).

### 6. Use the Calibration Results

The calibration results are saved in the output directory (default: `calibration_results_test/`):

1. **camera_params.json**: Contains all calibration parameters in JSON format.
2. **stereo_calibration.xml**: Contains calibration parameters in OpenCV XML format.
3. **rectification_map.png**: Visualization of rectified stereo images.
4. **original_pair.png**: Original stereo image pair for comparison.
5. **undistorted_samples/**: Directory containing undistorted sample images.

## Troubleshooting

If you encounter issues during calibration:

1. **High Calibration Error**:
   - Ensure the Charuco board is fully visible in all images.
   - Try capturing more images from different angles and distances.
   - Check if the board dimensions are correctly specified.

2. **Detection Failures**:
   - Ensure good lighting conditions without glare.
   - Make sure the board is not too far from the cameras.
   - Check if the board is printed correctly with high contrast.

3. **Unreasonable Camera Parameters**:
   - Verify that the camera resolution is correctly specified.
   - Check if the cameras are properly focused.
   - Ensure the cameras are not moving during image capture.

## Advanced Usage

For advanced users who want to customize the calibration process:

1. **Modify Calibration Parameters**:
   - Edit the configuration parameters in `calibration_wizard_improved_fixed.py`.
   - Adjust validation criteria for camera parameters if needed.

2. **Debug Charuco Detection**:
   - Use `debug_charuco_detection.py` to visualize the detection process.

3. **Visualize Rectification**:
   - Use `visualize_rectification.py` to examine the rectification results in detail.

## References

1. OpenCV Documentation: https://docs.opencv.org/
2. Charuco Board Tutorial: https://docs.opencv.org/master/df/d4a/tutorial_charuco_detection.html
3. Stereo Calibration Tutorial: https://docs.opencv.org/master/d9/d0c/group__calib3d.html
