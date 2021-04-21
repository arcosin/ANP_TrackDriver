import time, cv2, sys
from picamera import PiCamera
import numpy as np
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
from scripts import boost_contrast, resize

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
        self.width=256
        self.height=256
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
        self.size = self.width if self.width < self.height else self.height
        self.args['brightness'] = 0
        self.args['contrast'] = 0
        '''---------------------------------------------------'''
        time.sleep(5) # Wait for lens to warm up
        self.counter = 0
        
    def takePic(self):
        '''
            Create buffer and stores image into the array
            this is required process to calculate the image
        '''
        self.args['image'] = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.cam.capture(self.args['image'], 'rgb') #stores frame rgb info to np array
        '''---------------------------------------------------'''
        '''
            preprocess the np array by size, brightness, contrast
        '''
        #self.args['image'] = resize(self.args['image'], self.size)
        self.args['image'] = boost_contrast(**self.args) #preprocess the np array
        self.counter += 1
        return self.args['image']
        
            
    def destroy(self):
        self.cam.close()


