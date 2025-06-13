@echo off
echo Improved Stereo Camera Calibration Wizard (Fixed Version)
echo ================================================
echo.

REM Activate the virtual environment
call .\charuco\Scripts\activate

REM Run the improved calibration wizard with specified parameters
python calibration_wizard_improved_fixed.py --input_dir input_images --output_dir calibration_results_test --squares_x 9 --squares_y 7 --square_length 0.02 --marker_length 0.016 --baseline 0.1 --dictionary DICT_6X6_250

REM Pause to see the results
pause
