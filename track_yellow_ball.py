#!/usr/bin/env python3

import cv2
import numpy as np
import time

# Adafruit PCA9685 Imports
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# =========================================
# SERVO CONFIGURATION
# =========================================
SERVO_CHANNEL = 0    # PCA9685 output channel where your servo is connected
SERVO_MIN = 103      # 0 degrees (~500 µs)
SERVO_MAX = 512      # 180 degrees (~2500 µs)

# Initialize the servo at 90° (center)
servo_angle = 90

# =========================================
# PCA9685 INITIALIZATION
# =========================================
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50   # Typical servo frequency (50 Hz)

def set_servo_angle(channel, angle):
    """
    Moves the servo (on given PCA9685 'channel') to the specified 'angle' (0-180°)
    by mapping to the servo's min/max pulse range.
    """
    # Clamp angle to [0, 180]
    angle = max(0, min(180, angle))
    
    # Map angle to 12-bit pulse range for old Adafruit examples
    pulse_range = SERVO_MAX - SERVO_MIN
    pulse = SERVO_MIN + int((angle / 180.0) * pulse_range)
    
    # Convert from [0..4095] range to the 16-bit [0..65535] range used by the PCA9685 library
    duty_16bit = int((pulse / 4095.0) * 65535)
    pca.channels[channel].duty_cycle = duty_16bit

# Move servo to the initial 90° position
set_servo_angle(SERVO_CHANNEL, servo_angle)
time.sleep(1.0)  # Give it a moment to settle

# =========================================
# COLOR DETECTION CONFIG
# =========================================
# Adjust these HSV ranges for your lighting conditions / shade of yellow
LOWER_YELLOW = np.array([20, 100, 100], dtype=np.uint8)
UPPER_YELLOW = np.array([30, 255, 255], dtype=np.uint8)

# =========================================
# CAMERA SETUP
# =========================================
cap = cv2.VideoCapture(0)  # Use index 0 or another index if you have multiple cameras
if not cap.isOpened():
    print("Error: Could not open camera.")
    pca.deinit()
    exit(1)

# Optionally, you can reduce resolution if performance is an issue
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read one frame to confirm capture works
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    cap.release()
    pca.deinit()
    exit(1)

# =========================================
# SIMPLE "PD" CONTROL PARAMS
# =========================================
kp = 0.01            # Proportional gain (try 0.01 - 0.05)
deadband = 150        # Pixel error around center to ignore
use_derivative = True
kd = 0.01            # Derivative gain (start low, e.g., 0.005 - 0.02)
last_error_x = 0.0
last_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for yellow
        mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

        # Morphological operations (optional), to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Ignore tiny areas (noise)
            if area > 100:
                # Compute centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Draw contour/centroid for debug
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

                    # =========================================
                    # PD CONTROL TO MOVE SERVO
                    # =========================================
                    frame_center_x = frame.shape[1] // 2
                    error_x = cX - frame_center_x

                    # Deadband: if error is small, do nothing
                    if abs(error_x) > deadband:
                        # Proportional term
                        p_correction = kp * error_x

                        # Derivative term (optional)
                        if use_derivative:
                            current_time = time.time()
                            dt = current_time - last_time if last_time else 0
                            if dt > 0:
                                dError_x = (error_x - last_error_x) / dt
                                d_correction = kd * dError_x
                            else:
                                d_correction = 0

                            servo_angle -= (p_correction + d_correction)

                            last_time = current_time
                            last_error_x = error_x
                        else:
                            # Just P-control
                            servo_angle -= p_correction

                        # Command the servo
                        set_servo_angle(SERVO_CHANNEL, servo_angle)
        time.sleep(0.1)
        # Show windows for debugging
#        cv2.imshow('Frame', frame)
#        cv2.imshow('Mask', mask)

        # Press 'q' to exit
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

except KeyboardInterrupt:
    pass

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()

