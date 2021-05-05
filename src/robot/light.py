import RPi.GPIO as GPIO
import time

class Headlight:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(13, GPIO.OUT)
        GPIO.setup(6, GPIO.OUT)

    def switch(self, port, status):
        if port == 1:
            if status == 1:
                GPIO.output(13, GPIO.HIGH)
            elif status == 0:
                GPIO.output(13,GPIO.LOW)
            else:
                pass
        elif port == 2:
            if status == 1:
                GPIO.output(6, GPIO.HIGH)
            elif status == 0:
                GPIO.output(6,GPIO.LOW)
            else:
                pass
        else:
            print('Wrong Command: Example--switch(3, 1)->to switch on port3')

    def switch_off(self):
        self.switch(1,0)
        self.switch(2,0)

    def switch_on(self):
        self.switch(1, 1)
        self.switch(2, 1)

