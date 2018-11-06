
"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import numpy as np
import os
import shutil
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


FLAGS = None
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'input'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """ Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5]. """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels




class DataGenerator:
    
    def __init__(self, config):
        self.config = config

        # Get the data.
        self.train_data_filename   = maybe_download('train-images-idx3-ubyte.gz')
        self.train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        self.test_data_filename    = maybe_download('t10k-images-idx3-ubyte.gz')
        self.test_labels_filename  = maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into np arrays.
        self.train_data   = extract_data  (self.train_data_filename,   60000)
        self.train_labels = extract_labels(self.train_labels_filename, 60000)
        self.test_data    = extract_data  (self.test_data_filename,    10000)
        self.test_labels  = extract_labels(self.test_labels_filename,  10000)

        # Generate a validation set.
        self.validation_data   = self.train_data[:VALIDATION_SIZE, ...]
        self.validation_labels = self.train_labels[:VALIDATION_SIZE]
        self.train_data   = self.train_data[VALIDATION_SIZE:, ...]
        self.train_labels = self.train_labels[VALIDATION_SIZE:]
        self.num_epochs = NUM_EPOCHS
        self.train_size = self.train_labels.shape[0]
    
    def next_batch(self, batch_size):
        idx = np.random.choice(self.train_size, batch_size)
        yield self.train_data[idx], self.train_labels[idx]
