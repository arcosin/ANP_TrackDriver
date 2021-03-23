from time import sleep
from datetime import datetime
import picamera
import sys, os

DEFAULT_OUTDIR = '../images/'

def readCommand(argv):
  from optparse import OptionParser
  usageStr = """
    PURPOSE:  Captures image and saves to {0}
    USAGE:    python cap.py <options>
    EXAMPLES: (1) python cap.py
                  - Takes a picture after 5 seconds to {0}
              (2) python cap.py --time 10 --name <name>
                  - Takes a picture after 10 seconds, saves to {0}<name>.jpg
              (3) python cap.py -k
                  - Takes picture after waiting for keypress
  """.format(DEFAULT_OUTDIR)
  parser = OptionParser(usageStr)

  parser.add_option('-t', '--time', dest='time', type='int',
                    help=default('Delay in seconds'), default=5)
  parser.add_option('-n', '--name', dest='name',
                    help=default('Name of output file'),
                    default=str(datetime.now()))
  parser.add_option('-k', action='store_true', dest='should_wait',
                    help=default('Wait for keypress before taking picture'),
                    default=False)

  options, junk = parser.parse_args(argv)
  if len(junk) != 0:
    raise Exception('Command line input not understood: ' + str(junk))
  args = dict()

  args['time'] = options.time
  args['name'] = options.name
  args['should_wait'] = options.should_wait

  return args


def default(str):
  return str + ' [Default: %default]'


def record(time, name, should_wait):
  with picamera.PiCamera() as camera:
    camera.resolution = (2592,1944)
    camera.start_preview(fullscreen=False, window=(100, 20, 640, 480))
    if should_wait:
      print('Press enter to take picture')
      input()
    else:
      for i in range(time, 0, -1):
        print(i)
        sleep(1)
    camera.capture('%s.jpg' % (DEFAULT_OUTDIR + name))
    camera.stop_preview()


if __name__ == '__main__':
  args = readCommand(sys.argv[1:])
  os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
  record(**args)

