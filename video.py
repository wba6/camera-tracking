import cv2
from flask import Flask, Response
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit

######################################
# Servo initialization
######################################
kit = ServoKit(channels=16)
servo_channel = 0
kit.servo[servo_channel].set_pulse_width_range(500, 2500)

def set_servo_angle(angle, channel=0):
    """Set the servo angle within [0, 180]."""
    if 0 <= angle <= 180:
        kit.servo[channel].angle = angle
        print(f"Servo on channel {channel} set to angle {angle}")
    else:
        print("Angle must be between 0 and 180 degrees.")

# Starting servo angle
position = 90
set_servo_angle(position)

######################################
# YOLO initialization
######################################
# Load the YOLO model (example: default YOLOv8n on COCO)
model = YOLO("yolo11n_ncnn_model", task='detect')

######################################
# Camera initialization
######################################
cap = cv2.VideoCapture(0)

# Deadzone and movement speed
deadzone = 50
servoMovementSpeed = 2.0

app = Flask(__name__)

def gen_frames():
    """
    Generator function for streaming frames with bounding box drawn 
    around the largest detected person and adjusting the servo angle 
    to keep the person centered horizontally.
    """
    global position

    # Read one initial frame to get center
    success, frame = cap.read()
    if not success:
        print("Camera not found or unable to read from camera.")
        return

    rows, cols, _ = frame.shape
    center = cols // 2  # Horizontal center of the frame

    while True:
        success, frame = cap.read()
        if not success:
            continue  # Skip if frame is not read correctly

        # Run YOLO inference on the current frame
        results = model(frame, verbose=False)
        
        # Extract detection boxes
        detections = results[0].boxes
        person_boxes = []

        # Filter out only 'person' class. COCO 'person' class is index 0.
        for det in detections:
            class_id = int(det.cls.item())
            conf = det.conf.item()
            if class_id == 0 and conf > 0.5:  # Person class with confidence > 0.5
                # Convert YOLO detection (xyxy) to integer coords
                xyxy_tensor = det.xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                width = xmax - xmin
                height = ymax - ymin
                person_boxes.append((xmin, ymin, xmax, ymax, width * height))

        # If at least one person was detected, pick the largest bounding box
        if person_boxes:
            # Sort boxes by area (width*height)
            person_boxes.sort(key=lambda x: x[4], reverse=True)
            xmin, ymin, xmax, ymax, _ = person_boxes[0]  # Largest box by area
            # Draw bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Compute horizontal center of the largest-person bounding box
            x_medium = int((xmin + xmax) / 2)
        else:
            # If no person is detected, do not move servo
            x_medium = center
        
        # Draw a line at the bounding box center
        cv2.line(frame, (x_medium, 0), (x_medium, rows), (0, 255, 0), 2)

        # Update servo position if x_medium is outside the deadzone
        if x_medium < center - deadzone:
            position += servoMovementSpeed
        elif x_medium > center + deadzone:
            position -= servoMovementSpeed

        # Ensure position is clamped within [0, 180]
        position = max(min(position, 180), 0)
        set_servo_angle(position)

        # Encode the frame and stream via multipart HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Returns the video feed as a multipart/x-mixed-replace stream."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Disable debug for production; set debug=True for development
    app.run(host='0.0.0.0', debug=False, port=5000)

