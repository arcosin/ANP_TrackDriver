from time import sleep
from datetime import datetime
import picamera
import sys, os

DEFAULT_OUTDIR = '../videos/'

def readCommand(argv):
  from optparse import OptionParser
  usageStr = """
    PURPOSE:  Records video to {0}
    USAGE:    python rec.py <options>
    EXAMPLES: (1) python rec.py
                  - Records a 5 second video to {0}
              (2) python rec.py --time 10 --name <name>
                  - Records a 10 second video to {0}<name>.h264
              (3) python rec.py -k
                  - Begins recording for 5 seconds after keypress
  """.format(DEFAULT_OUTDIR)
  parser = OptionParser(usageStr)

  parser.add_option('-t', '--time', dest='time', type='int',
                    help=default('Time of recording in seconds'), default=5)
  parser.add_option('-n', '--name', dest='name',
                    help=default('Path to output location'),
                    default=str(datetime.now()))
  parser.add_option('-k', action='store_true', dest='should_wait',
                    help=default('Wait for keypress before recording'),
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
    camera.start_preview(fullscreen=False, window=(100,20,640,480))
    if should_wait:
      print('Press enter to begin recording')
      input()
      print('Recording...')
    camera.start_recording('%s.h264' % (DEFAULT_OUTDIR + name))
    sleep(time)
    camera.stop_recording()
    camera.stop_preview()


if __name__ == '__main__':
  args = readCommand(sys.argv[1:])
  os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
  record(**args)

