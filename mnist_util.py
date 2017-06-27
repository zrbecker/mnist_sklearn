import struct
import numpy as np

def mnist_read_images(filename):
    with open(filename, 'rb') as f:
        bytes = f.read()
    magic, count, rows, cols = struct.unpack_from('>iiii', bytes)
    if magic != 2051:
        raise Exception('failed to read mnist images from {}'.format(filename))
    images = np.frombuffer(bytes, dtype='uint8', offset=16)
    images = images.reshape(count, rows, cols)
    return images

def mnist_one_hot_encode(labels):
    max_label = max(labels)
    one_hot_encoding = []
    for label in labels:
        encoding = [0] * (max_label + 1)
        encoding[label] = 1
        one_hot_encoding.append(encoding)
    return np.array(one_hot_encoding).reshape(len(labels), max_label + 1)

def mnist_read_labels(filename, one_hot_encoding=False):
    with open(filename, 'rb') as f:
        bytes = f.read()
    magic, count = struct.unpack_from('>ii', bytes)
    if magic != 2049:
        raise Exception('failed to read mnist labels from {}'.format(filename))

    labels = np.frombuffer(bytes, dtype='uint8', offset=8)
    if one_hot_encoding:
        return mnist_one_hot_encode(labels)
    else:
        return labels

def mnist_read(images_filename, labels_filename, one_hot_encoding=False):
    images = mnist_read_images(images_filename)
    labels = mnist_read_labels(labels_filename, one_hot_encoding)
    return images, labels
