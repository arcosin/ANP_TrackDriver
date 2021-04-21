import RPi.GPIO as GPIO
import time

IR_sensor_right = 19
IR_sensor_middle = 16
IR_sensor_left = 20

class LineTracker:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_sensor_right, GPIO.IN)
        GPIO.setup(IR_sensor_middle, GPIO.IN)
        GPIO.setup(IR_sensor_left, GPIO.IN)

    def detect(self):
        status_right = GPIO.input(IR_sensor_right)
        status_middle = GPIO.input(IR_sensor_middle)
        status_left = GPIO.input(IR_sensor_left)

        line_detected = False
        if status_left or status_middle or status_right:
            line_detected = True
        
        return line_detected, status_right, status_middle, status_left
