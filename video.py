import cv2
from flask import Flask, Response
import numpy as np
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
servo_channel = 0
kit.servo[servo_channel].set_pulse_width_range(500,2500)

def set_servo_angle(angle, channel=0):
    if 0 <= angle <= 180:
        kit.servo[channel].angle = angle
        print(f"Servo on channel {channel} set to angle {angle}")
    else:
        print("Angle must be between 0 and 180 degrees.")

cap = cv2.VideoCapture(0)
position = 90
deadzone = 40
servoMovementSpeed = 1.5

set_servo_angle(90)

app = Flask(__name__)

def gen_frames():
    global position
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    center = cols // 2
    x_medium = center

    while True:
        success, frame = cap.read()
        if not success:
            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = np.array([161,155,84])
        high_red = np.array([179,255,255])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y),(x + w, y+h), (0,255,0),2)
            x_medium = int((x + x + w) / 2)
            break

        cv2.line(frame, (x_medium,0), (x_medium, rows), (0,255,0),2)

        if x_medium < center - deadzone:
            position += servoMovementSpeed
        elif x_medium > center + deadzone:
            position -= servoMovementSpeed
        set_servo_angle(position)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
