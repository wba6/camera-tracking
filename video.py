import cv2
from flask import Flask, Response, request
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
model = YOLO("yolo11n_ncnn_model", task='detect')
# model.names => {class_id: class_name} if it's a standard YOLO model

######################################
# Camera initialization
######################################
cap = cv2.VideoCapture(0)

# Deadzone and movement speed
deadzone = 60
servoMovementSpeed = 2.0

######################################
# Flask App
######################################
app = Flask(__name__)

# Global flags
override = False       # If True => manual control
classify_all = False   # If True => draw bounding boxes for all classes

@app.route('/')
def index():
    """
    Serve an HTML page with:
      - Checkboxes for Override and Classify All
      - A slider for servo angle
      - The video feed with click-and-drag logic (reversed)
    """
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Camera Control</title>
  <style>
    body {
      background-color: #f5f5f5;
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 900px;
      margin: 40px auto;
      background-color: #ffffff;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
      text-align: center;
      margin-top: 0;
    }
    #videoFeed {
      display: block;
      margin: 0 auto 20px auto;
      border: 2px solid #333;
      width: 640px;
      height: 480px;
      cursor: crosshair;
      border-radius: 4px;
    }
    .checkbox-bar {
      display: flex;
      justify-content: center;
      align-items: center;
      margin-bottom: 20px;
      gap: 40px;
    }
    .checkbox-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 16px;
    }
    .slider-container {
      text-align: center;
      margin-top: 20px;
    }
    .slider-container input[type="range"] {
      width: 300px;
      margin: 10px 0;
    }
    .footer {
      text-align: center;
      margin-top: 30px;
      font-size: 14px;
      color: #999;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Camera Control</h1>
    <img id="videoFeed" src="/video_feed" alt="Live Video Feed" />

    <div class="checkbox-bar">
      <div class="checkbox-item">
        <input type="checkbox" id="overrideCheckbox" />
        <label for="overrideCheckbox">Override servo control</label>
      </div>
      <div class="checkbox-item">
        <input type="checkbox" id="classifyAllCheckbox" />
        <label for="classifyAllCheckbox">Classify all objects</label>
      </div>
    </div>

    <h3>Manual Servo Control (Slider)</h3>
    <div class="slider-container">
      <input type="range" min="0" max="180" value="90" id="servoSlider" />
      <div>Slider Value (Degrees): <span id="servoAngleLabel">90</span></div>
    </div>
  </div>

  <script>
    // --------- CHECKBOX LOGIC -----------
    const overrideCheckbox = document.getElementById("overrideCheckbox");
    const classifyAllCheckbox = document.getElementById("classifyAllCheckbox");

    // On page load, we might want to fetch the current states from server
    // but for simplicity, let's assume they're OFF by default unless user toggles them.
    // If you want to get the real server state on load, you'd do a fetch to an endpoint
    // that returns override / classify_all in JSON, then set the checkbox states accordingly.

    // When user toggles "Override servo control"
    overrideCheckbox.addEventListener("change", () => {
      if (overrideCheckbox.checked) {
        // send /override_on
        fetch('/override_on', { method: 'POST' });
      } else {
        // send /override_off
        fetch('/override_off', { method: 'POST' });
      }
    });

    // When user toggles "Classify all objects"
    classifyAllCheckbox.addEventListener("change", () => {
      if (classifyAllCheckbox.checked) {
        // send /classify_all_on
        fetch('/classify_all_on', { method: 'POST' });
      } else {
        // send /classify_all_off
        fetch('/classify_all_off', { method: 'POST' });
      }
    });

    // ---------- SLIDER CONTROL -----------
    const slider = document.getElementById("servoSlider");
    const angleLabel = document.getElementById("servoAngleLabel");

    slider.addEventListener("input", function() {
      angleLabel.textContent = slider.value;
      fetch('/set_angle?angle=' + slider.value, { method: 'POST' });
    });

    // ---------- MOUSE DRAG CONTROL (Reversed) -----------
    let isDragging = false;
    let lastX = 0;
    const SERVO_MOVE_SCALE = 0.1;
    const videoFeed = document.getElementById("videoFeed");

    videoFeed.addEventListener("mousedown", function(event) {
      if (!overrideCheckbox.checked) return; // only works if override is ON
      isDragging = true;
      lastX = event.clientX;
    });

    videoFeed.addEventListener("mousemove", function(event) {
      if (!overrideCheckbox.checked || !isDragging) return;
      const currentX = event.clientX;
      const deltaX = currentX - lastX;
      lastX = currentX;

      // Reversed direction => position -= delta
      fetch('/adjust_servo?delta=' + (deltaX * SERVO_MOVE_SCALE), { method: 'POST' });
    });

    videoFeed.addEventListener("mouseup", function() {
      isDragging = false;
    });

    videoFeed.addEventListener("mouseleave", function() {
      isDragging = false;
    });
  </script>
</body>
</html>
'''

@app.route('/override_on', methods=['POST'])
def override_on():
    global override
    override = True
    return "Override is now ON. YOLO tracking disabled."

@app.route('/override_off', methods=['POST'])
def override_off():
    global override
    override = False
    return "Override is now OFF. YOLO tracking enabled."

@app.route('/classify_all_on', methods=['POST'])
def classify_all_on():
    global classify_all
    classify_all = True
    return "Classify All is now ON."

@app.route('/classify_all_off', methods=['POST'])
def classify_all_off():
    global classify_all
    classify_all = False
    return "Classify All is now OFF."

@app.route('/set_angle', methods=['POST'])
def set_angle():
    """
    Set the servo angle from slider input,
    reversed logic => servo_position = 180 - slider_value
    """
    global position
    angle_str = request.args.get('angle', default='90')
    try:
        angle = float(angle_str)
        angle = max(min(angle, 180), 0)
        # Reversed logic
        new_pos = 180 - angle
        position = new_pos
        set_servo_angle(position)
        return f"Slider angle={angle}, servo position={position}"
    except ValueError:
        return "Invalid angle", 400

@app.route('/adjust_servo', methods=['POST'])
def adjust_servo():
    """
    Adjust servo angle by a delta from mouse drag,
    reversed direction => position -= delta
    """
    global position
    delta_str = request.args.get('delta', default='0')
    try:
        delta = float(delta_str)
    except ValueError:
        return "Invalid delta", 400

    position -= delta  # reversed direction
    position = max(0, min(180, position))
    set_servo_angle(position)
    return f"Servo moved by -{delta}, new angle={position}"

def gen_frames():
    """
    Video streaming generator:
    - If classify_all=TRUE => draw bounding boxes for all classes (conf>0.5).
    - If classify_all=FALSE => draw bounding boxes only for persons.
    - Always track the largest person if override=FALSE.
    """
    global position, override, classify_all

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

            # If classify_all=ON => draw bounding boxes for everything.
            # If classify_all=OFF => only draw bounding boxes for person.
            if classify_all or (class_id == 0):
                if class_id == 0:
                    color = (0, 255, 0)  # green for person
                else:
                    color = (72, 72, 255) # other classes

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                class_name = model.names.get(class_id, f"id:{class_id}")
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # If it's a person, keep track for servo
            if class_id == 0:
                person_boxes.append((xmin, ymin, xmax, ymax, area))

        # If any persons found, pick largest
        if person_boxes:
            person_boxes.sort(key=lambda x: x[4], reverse=True)
            xmin, ymin, xmax, ymax, _ = person_boxes[0]
            x_medium = (xmin + xmax) // 2
        else:
            x_medium = center

        # Draw line for debugging
        cv2.line(frame, (x_medium, 0), (x_medium, rows), (0, 255, 0), 2)

        # Track only if override=FALSE
        if not override:
            if x_medium < center - deadzone:
                position += servoMovementSpeed
            elif x_medium > center + deadzone:
                position -= servoMovementSpeed
            position = max(min(position, 180), 0)
            set_servo_angle(position)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Returns the video feed as a multipart/x-mixed-replace stream."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

