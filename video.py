import cv2
import numpy as np
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# servo on channel zero
servo_channel = 0

#set servo pulse width range 
kit.servo[servo_channel].set_pulse_width_range(500,2500)

# function to set the angle of the servo
def set_servo_angle(angle, channel=0):
    if 0 <= angle <= 180:
        kit.servo[channel].angle = angle
        print(f"Servo on channel {channel} set to angle {angle}")
    else:
        print("Angle must be between 0 and 180 degrees.")

cap = cv2.VideoCapture(0)

#defualt x_medium
_, frame = cap.read()
rows,cols, _, = frame.shape
x_medium = int(cols/2)

# set center of the screen
center = int(cols/2)

#degrees
position = 90

# pixels
deadzone = 40

# degrees
servoMovementSpeed = 1.5

# defualt angle 
set_servo_angle(90)
while True:
    _, frame = cap.read()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #red color 
    low_red = np.array([161,155,84])
    high_red = np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)

    # get the red objects
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours so biggest is first
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)


    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)

        cv2.rectangle(frame, (x,y),(x + w, y+h), (0,255,0),2)

        # find the center line of the biggest object
        x_medium = int((x + x + w) / 2)
        break

    cv2.imshow("Frame", frame)

    # move servo left or right to follow object
    if x_medium < center - deadzone:
        position += servoMovementSpeed
    elif x_medium > center + deadzone:
        position -= servoMovementSpeed
    set_servo_angle(position)

cap.release()
c2.destroyAllWindows()
