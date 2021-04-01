import time
import drivetrain
import camera
import linetracker

def testDrivetrain():
    try:
        print("Starting")
        dt = drivetrain.DriveTrain()
        cam = camera.Camera()
        cam.takePic()

        print("Going fast")
        dt.moveSpeed(100, "forward")
        time.sleep(2)
        cam.takePic()

        print("Going left slow")
        dt.turnAngle(60)        
        dt.moveSpeed(100, "forward")
        time.sleep(2)
        cam.takePic()

        print("Stopping")
        dt.moveSpeed(0, "forward")
        dt.turnAngle(-60)
        
        #dt.driveHalt()
        dt.destroy()
        cam.destroy()
    # Allow ctrl-C-ing if something goes wrong
    except KeyboardInterrupt:
        dt.destroy()
        print("Terminated")

if __name__ == "__main__":
    #testDrivetrain()
    lt = linetracker.LineTracker()
    while True:
        try:
            if lt.detect()[0] == True:
                print("Detected")
            else:
                print("Not detected")
            
            time.sleep(1)
        except KeyboardInterrupt:
            print("Terminated")
            exit() 
