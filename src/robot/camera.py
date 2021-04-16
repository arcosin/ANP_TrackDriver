import time, cv2, sys
from picamera import PiCamera
import numpy as np
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
from scripts import preprocess

class Camera():
    '''
        https://picamera.readthedocs.io/en/release-1.12/fov.html#camera-modes
        Good to choose mod with binning (Binning increase the readout speed)
        Each mod have the minimal resolution recieved from the camera
        FoV represent the range of image captured from the camera
        If possible it is best to use Full FoV, 2x2 or 4x4 binning (higher = faster), mimimum resolution higher than requirement
        frame rates are also limited by each mode
        Please check if the camera is v1 or v2
    '''
    def __init__(self):
        self.width=640
        self.height=480
        while True:
            try:
                self.cam = PiCamera()
                self.cam.sensor_mode = 7
                self.framerate=60
                self.cam.resolution = (self.width, self.height)
                break
            except:
                print("Camera __init__ failed")
                print("Camera __init__ retry")
                pass
        self.args = dict()
        '''
            below three arguments could be commented
            to use default values of preprocessing file
        '''
        self.args['size'] = self.width if self.width < self.height else self.height
        self.args['brightness'] = 0
        self.args['contrast'] = 0
        '''---------------------------------------------------'''
        time.sleep(5) # Wait for lens to warm up
        self.counter = 0
        
    def takePic(self):
        '''
            below two arguments are required
            images will not be created without these arugments
        '''
        self.args['path'] = "img" + str(self.counter) + ".jpg"
        self.args['image'] = np.empty((self.height, self.width, 3), dtype=np.uint8)
        '''---------------------------------------------------'''
        self.cam.capture(self.args['image'], 'rgb') #stores frame rgb info to np array
        preprocess(**self.args) #preprocess the np array and save it as image
        self.counter += 1
            
    def destroy(self):
        self.cam.close()


