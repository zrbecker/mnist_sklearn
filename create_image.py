import argparse
import math
import numpy as np
from PIL import Image

from mnist_util import mnist_read_images

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, default='data/mnist_input_data.png')
parser.add_argument('--cols', type=int, default=64)
parser.add_argument('--rows', type=int, default=36)
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--randomize', type=bool, default=False)
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
args = parser.parse_args()

images = mnist_read_images(args.input)
if args.cols * args.rows > len(images):
    args.rows = math.ceil(count / args.cols)

if args.randomize:
    images_index_sample = np.random.choice(
            range(len(images)), args.cols * args.rows, replace=False)
else:
    images_index_sample = range(args.cols * args.rows)
images_sample = images[images_index_sample]

width = images.shape[2]
height = images.shape[1]
output_image = Image.new('L', (width * args.cols, height * args.rows))
for index, pixels in enumerate(images_sample):
    image = Image.fromarray(pixels, 'L')
    left = width * (index % args.cols)
    right = left + width
    upper = image.height * (index // args.cols)
    lower = upper + height
    output_image.paste(image, (left, upper, right, lower))

if args.resize:
    output_image = output_image.resize((args.width, args.height))
output_image.save(args.output)