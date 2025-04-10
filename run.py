# run.py

from routes import app
import config
import servo

if __name__ == '__main__':
    # Initialize the servo position before starting the app
    servo.init_servo(config.STARTING_SERVO_ANGLE)

    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False)
