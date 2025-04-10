# camera.py

import cv2
import numpy as np
from flask import Response
from yolo import model
import config
import servo

# Initialize camera
cap = cv2.VideoCapture(0)

def gen_frames():
    """
    Generator function that yields video frames in multipart/x-mixed-replace format.

    - If config.classify_all = True => draw bounding boxes for all classes (conf > 0.5).
    - If config.classify_all = False => draw bounding boxes for 'person' only (class_id = 0).
    - If config.override = False => automatically track the largest person.
    """
    # Test the camera once
    success, frame = cap.read()
    if not success:
        print("Camera not found or unable to read from camera.")
        return

    rows, cols, _ = frame.shape
    center = cols // 2

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Run YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        person_boxes = []

        for det in detections:
            class_id = int(det.cls.item())
            conf = det.conf.item()
            if conf < 0.5:
                continue

            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            area = (xmax - xmin) * (ymax - ymin)

            # Draw bounding boxes
            if config.classify_all or (class_id == 0):  # 0 => 'person' typically
                if class_id == 0:
                    color = (0, 255, 0)  # green for person
                else:
                    color = (72, 72, 255)  # some other color
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                class_name = model.names.get(class_id, f"id:{class_id}")
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Keep track of person boxes
            if class_id == 0:
                person_boxes.append((xmin, ymin, xmax, ymax, area))

        # If any persons found, pick largest
        if person_boxes:
            person_boxes.sort(key=lambda x: x[4], reverse=True)
            xmin, ymin, xmax, ymax, _ = person_boxes[0]
            x_medium = (xmin + xmax) // 2
        else:
            x_medium = center

        # Draw a vertical line at the target x
        cv2.line(frame, (x_medium, 0), (x_medium, rows), (0, 255, 0), 2)

        # If not override => auto-track
        if not config.override:
            if x_medium < center - config.DEADZONE:
                servo.position += config.SERVO_MOVEMENT_SPEED
            elif x_medium > center + config.DEADZONE:
                servo.position -= config.SERVO_MOVEMENT_SPEED

            # Clamp servo angle
            servo.position = max(0, min(180, servo.position))
            servo.set_servo_angle(servo.position)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

