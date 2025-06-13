# Charuco Stereo Camera Calibration System

This system provides tools for calibrating stereo camera setups using Charuco boards.

## **CAUTION!! New Updates are completely different, go to check out docs/HowToCalibrate.md for more details!.**



---











## Charuco Board Specifications

This calibration system is configured for a specific Charuco board with the following parameters:

| Parameter         | Value          | Description                                                  |
| ----------------- | -------------- | ------------------------------------------------------------ |
| **Target Type**   | ChArUco Board  | Combined chessboard corners with ArUco markers for robust detection |
| **Board Width**   | 210mm          | Physical width of the board (calculated based on columns and square size) |
| **Board Height**  | 297mm          | Physical height of the board (calculated based on rows and square size) |
| **Rows**          | 7              | Number of internal corner rows (corresponds to 8 squares vertically) |
| **Columns**       | 9              | Number of internal corner columns (corresponds to 10 squares horizontally) |
| **Square Size**   | 20mm           | Width of each chessboard square |
| **Dictionary**    | `DICT_6X6_250` | Standard ArUco dictionary to ensure marker uniqueness |

## Camera Setup

The stereo camera baseline (distance between camera centers) is fixed at 100mm.

## Setup

1. Activate the virtual environment:
   - Windows: `.\charuco\Scripts\activate`
   - Linux/Mac: `source charuco/bin/activate`

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Input Structure

Place your stereo image pairs in the following structure:
```
input_images/
├── left/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── right/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

Ensure that corresponding images in left and right folders have the same filename.

## Usage

### Generate a Charuco Board

```
python generate_charuco.py --output charuco_board.png
```

This will generate a Charuco board with the following specifications:
- 9 squares in X direction (columns)
- 7 squares in Y direction (rows)
- 20mm square size
- 15mm marker size
- DICT_6X6_250 dictionary

Print the generated board at its actual size (210mm x 297mm) for accurate calibration.

### Calibrate Stereo Cameras

```
python calibrate_stereo.py --input_dir input_images --output_dir calibration_results --baseline 0.1
```

The `--baseline` parameter specifies the known distance between camera centers in meters (0.1m = 100mm). This ensures accurate scale in the calibration results.

## Calibration Output

The calibration process generates the following files in the output directory:

- `camera_params.json`: Contains camera matrices, distortion coefficients, and stereo parameters
- `calibration_report.txt`: Human-readable report of calibration results
- `rectification_map.png`: Visualization of the rectification results
- `detections/`: Directory containing images with detected Charuco corners (if `--draw` flag is used)

## Interpreting Results

- **Camera Matrix**: Intrinsic parameters of each camera
- **Distortion Coefficients**: Parameters to correct lens distortion
- **Rotation Matrix**: Rotation between the left and right camera
- **Translation Vector**: Translation between the left and right camera
- **Baseline**: Distance between camera centers (fixed to 100mm)
- **Essential Matrix**: Relates corresponding points in stereo images
- **Fundamental Matrix**: Relates corresponding points in pixel coordinates
- **Rectification Matrices**: Rotations applied to align epipolar lines
- **Projection Matrices**: Projection from 3D to 2D for rectified images
- **Disparity-to-Depth Matrix**: Converts disparity to actual depth

## Visualization

To visualize the rectification results:

```
python visualize_rectification.py --input_dir input_images --params calibration_results/camera_params.json --output_dir rectification_results --anaglyph
```

This will generate:
- Rectified left and right images
- Side-by-side visualizations with horizontal epipolar lines
- Anaglyph (red-cyan) 3D images for viewing with 3D glasses

The `--anaglyph` flag is optional and creates red-cyan 3D images from the rectified stereo pairs.
