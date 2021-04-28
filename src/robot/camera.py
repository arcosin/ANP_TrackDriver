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
    def __init__(self, sensor_mode=7, framerate=60, width=512, height=256, brightness=64, contrast=64):
        self.width=width
        self.height=height
        while True:
            try:
                self.cam = PiCamera()
                self.cam.sensor_mode = sensor_mode
                self.framerate=framerate
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
        self.args['brightness'] = brightness
        self.args['contrast'] = contrast
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

if __name__ == "__main__":
    camera = Camera(sensor_mode=5, brightness=64, contrast=64)
    from PIL import Image
    im = Image.fromarray(camera.takePic())
    im.save('./5.jpg')
