#!/usr/bin/env python3

import time
import cv2
import numpy as np

# Adafruit PCA9685 Imports
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# ================================
# SERVO CONFIGURATION
# ================================
SERVO_CHANNEL = 0    # PCA9685 channel your servo is on
SERVO_MIN = 103      # 0 degrees (~500 µs)
SERVO_MAX = 512      # 180 degrees (~2500 µs)
servo_angle = 90     # Start servo at 90° (center)

# ================================
# PCA9685 SETUP
# ================================
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50   # Typical servo frequency (50 Hz)

def set_servo_angle(channel, angle):
    """
    Moves the servo on 'channel' to 'angle' in [0..180].
    Maps angle to the servo's pulse range, then sets the PCA9685 duty_cycle.
    """
    # Clamp angle
    angle = max(0, min(180, angle))

    # Map [0..180] -> [SERVO_MIN..SERVO_MAX]
    pulse_range = SERVO_MAX - SERVO_MIN
    pulse = SERVO_MIN + int((angle / 180.0) * pulse_range)

    # Convert [0..4095] -> [0..65535]
    duty_16bit = int((pulse / 4095.0) * 65535)
    pca.channels[channel].duty_cycle = duty_16bit

# Move servo to center
set_servo_angle(SERVO_CHANNEL, servo_angle)
time.sleep(1.0)

# ================================
# HAAR CASCADE FOR FACE DETECTION
# ================================
# OpenCV typically includes haarcascades in cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load haarcascade_frontalface_default.xml.")
    pca.deinit()
    exit(1)

# ================================
# CAMERA SETUP
# ================================
cap = cv2.VideoCapture(0)  # If you have multiple cams, adjust index
if not cap.isOpened():
    print("Error: Could not open camera.")
    pca.deinit()
    exit(1)

# Optional: reduce resolution if needed for speed
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Test read
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    cap.release()
    pca.deinit()
    exit(1)

# ================================
# PD CONTROL PARAMETERS
# ================================
kp = 0.01             # Proportional gain
kd = 0.01             # Derivative gain
deadband = 50        # Pixels around center to ignore
use_derivative = True

last_error_x = 0
last_time = time.time()

print("Tracking started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale for Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            # Pick largest face by area
            (x, y, w, h) = max(faces, key=lambda rect: rect[2]*rect[3])

            # Face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Draw a rectangle around face and center
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 3, (0, 0, 255), -1)

            # ================================
            # PD CONTROL FOR SERVO PANNING
            # ================================
            frame_center_x = frame.shape[1] // 2
            error_x = face_center_x - frame_center_x

            # Only move servo if error is outside deadband
            if abs(error_x) > deadband:
                # Proportional term
                p_correction = kp * error_x

                if use_derivative:
                    current_time = time.time()
                    dt = current_time - last_time if last_time else 0.0

                    if dt > 0:
                        dError_x = (error_x - last_error_x) / dt
                        d_correction = kd * dError_x
                    else:
                        d_correction = 0.0

                    servo_angle -= (p_correction + d_correction)

                    last_time = current_time
                    last_error_x = error_x
                else:
                    servo_angle -= p_correction

                set_servo_angle(SERVO_CHANNEL, servo_angle)
        time.sleep(0.1)
        # Show debug window
        cv2.imshow("Frame", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()

