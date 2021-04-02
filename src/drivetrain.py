# Ensures proper division mapping on / Needed for Adafruit lib
from __future__ import division 
import time
import sys

import RPi.GPIO as GPIO
import Adafruit_PCA9685

# References: 
# https://github.com/adafruit/Adafruit_Python_PCA9685/blob/master/examples/simpletest.py
# https://learn.adafruit.com/16-channel-pwm-servo-driver/library-reference
# https://howtomechatronics.com/how-it-works/how-servo-motors-work-how-to-control-servos-using-arduino/
# https://www.adeept.com/learn/tutorial-249.html
# https://www.adeept.com/learn/tutorial-252.html
# Understanding of the code in this file works can be gained by surfing the above links

class DriveTrain:
    def __init__(self):
        # Right now, we only use this to control the turn servo
        # on the drivetrain. Eventually, it will have to be moved to
        # a global/shared context to control the claw servos. 
        self.turn_pwm = Adafruit_PCA9685.PCA9685()
        # 50Hz PWM frequency => servo expects updates every 1/50Hz = 20ms
        self.turn_pwm.set_pwm_freq(50)   

        # Pin numbers for back wheels (forward/backward)
        self.motor_A_pin1 = 26
        self.motor_A_pin2 = 21
        self.motor_A_en = 4

        self.motor_B_pin1 = 27
        self.motor_B_pin2 = 18
        self.motor_B_en = 17

        # Just declarations
        self.motor_pwm_A = 0    
        self.motor_pwm_B = 0        

        # Constants for turning servo
        self.initPos = 300
        self.maxPos = 560
        self.minPos = 100
        self.angleRange = 180

        self.driveSetup()
        self.turnSetup()

    def driveSetup(self):
        #GPIO.setwarnings(False)

        #Broadcomm chip specific pin nums
        GPIO.setmode(GPIO.BCM) 
        GPIO.setup(self.motor_A_pin1, GPIO.OUT)
        GPIO.setup(self.motor_A_pin2, GPIO.OUT)
        GPIO.setup(self.motor_A_en, GPIO.OUT)
        GPIO.setup(self.motor_B_pin1, GPIO.OUT)
        GPIO.setup(self.motor_B_pin2, GPIO.OUT)
        GPIO.setup(self.motor_B_en, GPIO.OUT)

        self.driveHalt()

        #Enclose in try except pass if this don't work
        self.motor_pwm_A = GPIO.PWM(self.motor_A_en, 1000)
        self.motor_pwm_B = GPIO.PWM(self.motor_B_en, 1000)

    def driveHalt(self):
        GPIO.output(self.motor_A_pin1, GPIO.LOW)
        GPIO.output(self.motor_A_pin2, GPIO.LOW)
        GPIO.output(self.motor_A_en, GPIO.LOW)
        GPIO.output(self.motor_B_pin1, GPIO.LOW)
        GPIO.output(self.motor_B_pin2, GPIO.LOW)
        GPIO.output(self.motor_B_en, GPIO.LOW)

        self.turn_pwm.set_pwm(0, 0, self.initPos)

    def turnSetup(self, initPos = 300, moveTo = 1):
        if initPos > self.minPos and initPos < self.maxPos:
            if moveTo:
                # First arg is ID/channel of the motor - in this case 0
                self.turn_pwm.set_pwm(0, 0, initPos)
        else:
            strErrorMsg = "Drivetrain: Invalid input position" + str(initPos) + ", minPos = " + str(self.minPos) + ", maxPos = " + str(self.maxPos)
            print(strErrorMsg)

    def moveSpeed(self, speed, direction):
        # Correct combination of LOW/HIGH pin settings were found by lifting the bot
        # and trying until it worked as intended
        if direction == "backward":
            GPIO.output(self.motor_A_pin1, GPIO.LOW)
            GPIO.output(self.motor_A_pin2, GPIO.HIGH)
            self.motor_pwm_A.start(0)
            self.motor_pwm_A.ChangeDutyCycle(speed)

            GPIO.output(self.motor_B_pin1, GPIO.LOW)
            GPIO.output(self.motor_B_pin2, GPIO.HIGH)
            self.motor_pwm_B.start(0)
            self.motor_pwm_B.ChangeDutyCycle(speed)

        elif direction == "forward":
            GPIO.output(self.motor_A_pin1, GPIO.HIGH)
            GPIO.output(self.motor_A_pin2, GPIO.LOW)
            self.motor_pwm_A.start(100)
            self.motor_pwm_A.ChangeDutyCycle(speed)

            GPIO.output(self.motor_B_pin1, GPIO.HIGH)
            GPIO.output(self.motor_B_pin2, GPIO.LOW)
            self.motor_pwm_B.start(100)
            self.motor_pwm_B.ChangeDutyCycle(speed)

    def turnAngle(self, angle):
        # Positive input is left, negative input is right
        pwmOut = int((self.maxPos - self.minPos)/self.angleRange*angle)
        setPos = int(self.initPos + pwmOut)
        if setPos > self.maxPos: setPos = self.maxPos
        elif setPos < self.minPos: setPos = self.minPos

        self.turn_pwm.set_pwm(0, 0, setPos)

    def destroy(self):
        # Add logic for to uninstanitiate turn servo
        self.driveHalt()
        GPIO.cleanup()
