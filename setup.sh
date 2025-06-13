#!/bin/bash

echo "Setting up Charuco stereo calibration system..."

# Create virtual environment
python3 -m venv charuco

# Activate virtual environment
source charuco/bin/activate

# Install required packages
pip install -r requirements.txt

echo ""
echo "Setup complete! You can now use the Charuco stereo calibration system."
echo ""
echo "To activate the virtual environment, run:"
echo "  source charuco/bin/activate"
echo ""
echo "To generate a Charuco board, run:"
echo "  python generate_charuco.py --output charuco_board.png"
echo ""
echo "To calibrate stereo cameras, run:"
echo "  python calibrate_stereo.py --input_dir input_images --output_dir calibration_results"
echo ""
echo "To visualize rectification, run:"
echo "  python visualize_rectification.py --input_dir input_images --params calibration_results/camera_params.json"
echo ""

# Create input directories
mkdir -p input_images/left
mkdir -p input_images/right

echo "Input directories created:"
echo "  input_images/left"
echo "  input_images/right"
echo ""
echo "Place your stereo image pairs in these directories."
echo ""

# Make scripts executable
chmod +x generate_charuco.py
chmod +x calibrate_stereo.py
chmod +x visualize_rectification.py
