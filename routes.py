# routes.py

from flask import Flask, request, Response
import config
import servo
from camera import gen_frames

app = Flask(__name__)

@app.route('/')
def index():
    """
    Return a small HTML page with:
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

    overrideCheckbox.addEventListener("change", () => {
      if (overrideCheckbox.checked) {
        fetch('/override_on', { method: 'POST' });
      } else {
        fetch('/override_off', { method: 'POST' });
      }
    });

    classifyAllCheckbox.addEventListener("change", () => {
      if (classifyAllCheckbox.checked) {
        fetch('/classify_all_on', { method: 'POST' });
      } else {
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
      if (!overrideCheckbox.checked) return;
      isDragging = true;
      lastX = event.clientX;
    });

    videoFeed.addEventListener("mousemove", function(event) {
      if (!overrideCheckbox.checked || !isDragging) return;
      const currentX = event.clientX;
      const deltaX = currentX - lastX;
      lastX = currentX;
      // reversed => send -delta
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
    config.override = True
    return "Override is now ON. YOLO tracking disabled."

@app.route('/override_off', methods=['POST'])
def override_off():
    config.override = False
    return "Override is now OFF. YOLO tracking enabled."

@app.route('/classify_all_on', methods=['POST'])
def classify_all_on():
    config.classify_all = True
    return "Classify All is now ON."

@app.route('/classify_all_off', methods=['POST'])
def classify_all_off():
    config.classify_all = False
    return "Classify All is now OFF."

@app.route('/set_angle', methods=['POST'])
def set_angle():
    """
    Set the servo angle from slider input,
    reversed logic => servo_position = 180 - slider_value
    """
    angle_str = request.args.get('angle', default='90')
    try:
        angle = float(angle_str)
        angle = max(min(angle, 180), 0)
        # Reversed logic
        new_pos = 180 - angle
        servo.position = new_pos
        servo.set_servo_angle(servo.position)
        return f"Slider angle={angle}, servo position={servo.position}"
    except ValueError:
        return "Invalid angle", 400

@app.route('/adjust_servo', methods=['POST'])
def adjust_servo():
    """
    Adjust servo angle by a delta from mouse drag,
    reversed direction => position -= delta
    """
    delta_str = request.args.get('delta', default='0')
    try:
        delta = float(delta_str)
    except ValueError:
        return "Invalid delta", 400

    servo.position -= delta  # reversed direction
    servo.position = max(0, min(180, servo.position))
    servo.set_servo_angle(servo.position)
    return f"Servo moved by -{delta}, new angle={servo.position}"

@app.route('/video_feed')
def video_feed():
    """Returns the video feed as a multipart/x-mixed-replace stream."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

