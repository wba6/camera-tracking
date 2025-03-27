#!/usr/bin/env python3
import cv2
import time
import numpy as np
import Adafruit_PCA9685

# ------------------------------------------------
# 1. Servo and PCA9685 Setup for MG996R
# ------------------------------------------------
pwm = Adafruit_PCA9685.PCA9685(busnum=1)
pwm.set_pwm_freq(50)  # 50Hz for typical servos

SERVO_CHANNEL = 0
SERVO_MIN = 103   # Pulse count for 0° (~500µs)
SERVO_MAX = 512   # Pulse count for 180° (~2500µs)

# Global servo state (starting at center = 90°)
current_angle = 90

def set_servo_angle(angle):
    """Set the servo to a specified angle and update the current_angle."""
    global current_angle
    angle = max(0, min(180, angle))
    pulse = int(SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN))
    pwm.set_pwm(SERVO_CHANNEL, 0, pulse)
    current_angle = angle
    print(f"Setting angle to {angle:.1f}° (pulse: {pulse})")

# Initialize the servo at center.
set_servo_angle(current_angle)

# ------------------------------------------------
# 2. OpenCV DNN Face Detector Setup
# ------------------------------------------------
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

# Load the DNN face detector model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_faces_dnn(frame, conf_threshold=0.4):
    """
    Uses OpenCV's DNN module to detect faces.
    Returns a list of bounding boxes in (x, y, w, h) format.
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

# ------------------------------------------------
# 3. Global Parameters for Tracking and Scanning Modes
# ------------------------------------------------
CAMERA_FOV = 95.0         # Horizontal Field of View in degrees for the 2K camera
MIN_FACE_AREA = 500      # Minimum face area (in pixels) to consider valid

# Scanning mode parameters
scanning_step = 5         # Degrees per iteration when scanning
scan_direction = 1        # 1 for right, -1 for left

# Maximum servo movement per iteration (degrees)
MAX_DELTA = 5.0

# Deadband (degrees): if error is below this, do not adjust.
DEADBAND = 20.0

# Smoothing parameters for target angle filtering
smoothing_weight = 0.3    # Weight for new target; lower values make the change slower.
smoothed_target_angle = 90  # Initialize with the servo center

# ------------------------------------------------
# 4. Face Tracking Mode
# ------------------------------------------------
def face_tracking_mode(frame):
    """
    Uses the DNN face detector to find a face and compute a target angle.
    The target angle is smoothed over time to avoid oscillations.
    If the error is within the deadband, no correction is made.
    Returns (detected_flag, annotated_frame).
    """
    global current_angle, smoothed_target_angle
    faces = detect_faces_dnn(frame, conf_threshold=0.4)
    if len(faces) > 0:
        # Pick the largest detected face.
        (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
        face_area = w * h
        if face_area < MIN_FACE_AREA:
            print(f"Ignored detection with area {face_area}")
            return False, frame

        # Calculate centers.
        face_center_x = x + w / 2
        frame_width = frame.shape[1]
        frame_center_x = frame_width / 2

        # Convert pixel error to degrees.
        error_pixels = face_center_x - frame_center_x
        error_degrees = error_pixels * (CAMERA_FOV / frame_width)

        # Compute the immediate target angle (90° is center).
        target_angle = 90 - error_degrees

        # Exponentially smooth the target angle.
        smoothed_target_angle = (1 - smoothing_weight) * smoothed_target_angle + smoothing_weight * target_angle

        # Calculate the difference.
        diff = smoothed_target_angle - current_angle

        # If the difference is within the deadband, consider the face aligned.
        if abs(diff) < DEADBAND:
            print("Face detected and aligned.")
            cv2.putText(frame, "Face Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return True, frame

        # Adjust the servo: compute a delta and cap it.
        # Use a smoothing factor based on error magnitude.
        if abs(diff) > 20:
            alpha = 0.6
        elif abs(diff) > 10:
            alpha = 0.5
        else:
            alpha = 0.3

        raw_delta = alpha * diff
        delta = max(-MAX_DELTA, min(MAX_DELTA, raw_delta))
        new_angle = current_angle + delta
        new_angle = max(0, min(180, new_angle))
        set_servo_angle(new_angle)

        # Annotate the frame.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking Angle: {current_angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return True, frame
    else:
        return False, frame

# ------------------------------------------------
# 5. Scanning Mode
# ------------------------------------------------
def scanning_mode():
    """
    When no face is detected, the servo sweeps left and right.
    """
    global current_angle, scan_direction
    new_angle = current_angle + (scan_direction * scanning_step)
    if new_angle <= 0 or new_angle >= 180:
        scan_direction = -scan_direction
        new_angle = current_angle + (scan_direction * scanning_step)
    set_servo_angle(new_angle)

# ------------------------------------------------
# 6. Main Loop
# ------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detected, frame = face_tracking_mode(frame)
        if not detected:
            scanning_mode()
            cv2.putText(frame, "Scanning...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#        cv2.imshow("Face Tracking", frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    cap.release()
    cv2.destroyAllWindows()

