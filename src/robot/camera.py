import time
from picamera import PiCamera

class Camera():
    def __init__(self):
        # TODO: Protect with try except
        self.cam = PiCamera()
        self.cam.resolution = (640, 480)
        time.sleep(5)
        self.counter = 0
        
    def takePic(self):
        fileName = "img" + str(self.counter) + ".jpg"
        self.cam.capture(fileName)
        self.counter += 1
            
    def destroy(self):
        self.cam.close()


