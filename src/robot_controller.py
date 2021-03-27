import time
import drivetrain
import camera

if __name__ == "__main__":
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
