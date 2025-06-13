@echo off
echo Setting up Charuco stereo calibration system...

REM Activate virtual environment
call .\charuco\Scripts\activate

REM Install required packages
pip install -r requirements.txt

echo.
echo Setup complete! You can now use the Charuco stereo calibration system.
echo.
echo To activate the virtual environment, run:
echo   .\charuco\Scripts\activate
echo.
echo To generate a Charuco board, run:
echo   python generate_charuco.py --output charuco_board.png
echo.
echo To calibrate stereo cameras, run:
echo   python calibrate_stereo.py --input_dir input_images --output_dir calibration_results
echo.
echo To visualize rectification, run:
echo   python visualize_rectification.py --input_dir input_images --params calibration_results/camera_params.json
echo.

REM Create input directories
mkdir input_images\left
mkdir input_images\right

echo Input directories created:
echo   input_images\left
echo   input_images\right
echo.
echo Place your stereo image pairs in these directories.
echo.
