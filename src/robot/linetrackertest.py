import linetracker
import time

def main():
    lt = linetracker.LineTracker()
    print("press enter to display linetracker info, or q to quit")
    while (True):
        #opt =  input()
        #run = True
        #if (opt) == 'q': break
        time.sleep(0.125)
        detected, right, middle, left = lt.detect()
        print("%d: %d, %d, %d" % (detected, right, middle, left))

if __name__ == '__main__':
    main()
