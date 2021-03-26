"""
  This can be used as a command-line tool to preprocess images manually, or it can be
  imported into another file via `from preprocess import preprocess`
"""

import sys, os
import cv2
import numpy as np

DEFAULT_OUTDIR = '../images/preprocessed/'
DEFAULT_SIZE = 300
DEFAULT_BRIGHTNESS = 0
DEFAULT_CONTRAST = 64

def readCommand(argv):
  # Extract required first argument
  args = dict()
  if argv[0] != '-h' and argv[0] != '--help':
    args['image'] = argv[0]
    del argv[0]

  from optparse import OptionParser
  usageStr = """
    PURPOSE:  Resizes and boosts contrast of image
    USAGE:    python preprocess.py <image> <options>
    EXAMPLES: (1) python preprocess.py image.jpg
                  - Center crops to {1}x{1}, boosts contrast, saves to {0}
              (2) python preprocess.py image.jpg -b 127 -c 64 -o dir
                  - Center crops to {1}x{1}, boosts contrast with brightness 127 and contrast 64, saves to ./dir
    """.format(DEFAULT_OUTDIR, DEFAULT_SIZE)
  parser = OptionParser(usageStr)

  parser.add_option('-s', '--size', dest='size', type='int',
                    help=default('Size of output image'), default=DEFAULT_SIZE)
  parser.add_option('-b', '--brightness', dest='brightness', type='int',
                    help=default('Brightness value, range [-127, 127]'), default=DEFAULT_BRIGHTNESS)
  parser.add_option('-c', '--contrast', dest='contrast', type='int',
                    help=default('Contrast value, range [-127, 127]'), default=DEFAULT_CONTRAST)
  parser.add_option('-o', '--out', dest='outdir',
                    help=default('Output destination'),
                    default=os.path.join(DEFAULT_OUTDIR, os.path.basename(args.get('image', ''))))
  parser.add_option('--sample', action='store_true', dest='sample',
                    help=default('Output various different brightness/contrast combinations'),
                    default=False)

  options, junk = parser.parse_args(argv)
  if len(junk) != 0:
    raise Exception('Command line input not understood: ' + str(junk))

  args['image'] = cv2.imread(args['image'])
  args['size'] = options.size
  args['brightness'] = options.brightness
  args['contrast'] = options.contrast
  args['path'] = options.outdir
  args['sample'] = options.sample

  return args


def default(str):
  return str + ' [Default: %default]'


def preprocess(image, size=DEFAULT_SIZE, brightness=DEFAULT_BRIGHTNESS, contrast=DEFAULT_CONTRAST, path=None):
  res = boost_contrast(resize(image, size), brightness, contrast)
  if path:
    cv2.imwrite(path, res)
  return res


def boost_contrast(image, brightness=DEFAULT_BRIGHTNESS, contrast=DEFAULT_CONTRAST):
  if brightness != 0:
    if brightness > 0:
      shadow = brightness
      highlight = 255
    else:
      shadow = 0
      highlight = 255 + brightness
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow
    buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
  else:
    buf = image.copy()

  if contrast != 0:
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

  return buf


def resize(image, size=DEFAULT_SIZE):
  # Downsample, preserve aspect ratio
  scale = 30
  w = int(image.shape[1] * scale / 100)
  h = int(image.shape[0] * scale / 100)
  image = cv2.resize(image, (w, h), 0, 0, cv2.INTER_AREA)

  # Center crop to desired dimensions
  center = (image.shape[0] / 2, image.shape[1] / 2)
  x = int(center[1] - size / 2)
  y = int(center[0] - size / 2)
  return image[y:y+size, x:x+size]


def demo(image, path, size=DEFAULT_SIZE):
  image = resize(image)
  out = np.zeros((size * 2, size * 3, 3), dtype=np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fcolor = (0, 0, 0)
  brightness = [0, -127, 127,   0,  0, 64]
  contrast   = [0,    0,   0, -64, 64, 64]
  for i, b in enumerate(brightness):
    c = contrast[i]
    row = size * (i // 3)
    col = size * (i % 3)

    out[row:row + size, col: col + size] = boost_contrast(image, b, c)
    msg = 'b %d' % b
    cv2.putText(out, msg, (col, row + size - 22), font, .7, fcolor, 1, cv2.LINE_AA)
    msg = 'c %d' % c
    cv2.putText(out, msg, (col, row + size - 4), font, .7, fcolor, 1, cv2.LINE_AA)

  cv2.imwrite(path, out)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Missing required filename argument. See -h or --help for usage')
    exit()
  args = readCommand(sys.argv[1:])
  os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
  if args['sample']:
    demo(args['image'], args['path'], args['size'])
  else:
    preprocess(**args)

