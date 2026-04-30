
FYP Gimbal Code Package (updated with your Pixhawk servo controller)

FILES
1) servo_interface.py
   - real PixhawkServoController class using pymavlink
   - main function used by the rest of the pipeline:
       set_pan_tilt(pan_deg, pitch_deg)
   - also includes spray_action_stub(dwell_time_s)

2) manual_servo_test.py
   - simplified version of your teammates' interactive script
   - lets you type pan/tilt angles manually

3) section1_center_alignment.py
   - laser dot detection
   - image center extraction
   - center-alignment error

4) section2_calibration_matrix.py
   - collect calibration samples for pan/pitch sweeps
   - fit 2x2 calibration matrix

5) section3_coverage_kernel.py
   - estimate spray footprint kernel
   - generate cleaning points with overlap

6) section4_cleaning_serpentine.py
   - use A^(-1) to convert pixel targets to pan/pitch commands
   - execute serpentine cleaning pass

7) demo_pipeline.py
   - example showing how all pieces connect

HOW TO TEST ONLY THE SERVOS
cd into the folder and run:
    python3 manual_servo_test.py

WHY THIS VERSION IS CLEANER
- removes repeated code from the original script
- keeps the same PWM mapping and MAVLink command
- turns the servo logic into a reusable class
- lets the calibration and cleaning pipeline call the servo controller directly
