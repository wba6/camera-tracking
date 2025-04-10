# servo.py

from adafruit_servokit import ServoKit

# Initialize the ServoKit
kit = ServoKit(channels=16)

# Set your servo channel here
SERVO_CHANNEL = 0

# Configure the servo pulse widths if needed
kit.servo[SERVO_CHANNEL].set_pulse_width_range(500, 2500)

# Global variable to track servo's current angle
position = 0  # Will set properly in init_servo()

def init_servo(start_angle: int = 90):
    """Initialize the global servo angle and set it."""
    global position
    position = start_angle
    set_servo_angle(position)

def set_servo_angle(angle: float, channel: int = SERVO_CHANNEL):
    """Set the servo angle (0 to 180)."""
    if 0 <= angle <= 180:
        kit.servo[channel].angle = angle
        print(f"Servo on channel {channel} set to angle {angle}")
    else:
        print("Angle must be between 0 and 180 degrees.")
