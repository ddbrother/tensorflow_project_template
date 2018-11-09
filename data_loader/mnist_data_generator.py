
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

import keras
from keras.datasets import mnist


class DataGenerator:
    
    def __init__(self, config):
        self.config = config

        # Get the data, mnist.npz is in ~/.keras/datasets/mnist.npz
        print("Loading the MNIST data in ~/.keras/datasets/mnist.npz")
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = mnist.load_data()
        self.train_data   = self.train_data  .reshape(-1,28,28,1).astype(np.float32)
        self.train_labels = self.train_labels.reshape(-1)        .astype(np.int64)
        self.test_data   = self.test_data  .reshape(-1,28,28,1).astype(np.float32)
        self.test_labels = self.test_labels.reshape(-1)        .astype(np.int64)

        self.train_data = 2.0*self.train_data/self.config.pixel_depth - 1.0
        self.test_data  = 2.0*self.test_data /self.config.pixel_depth - 1.0

        print("train data shape:", self.train_data.shape)
        print("test  data shape:", self.test_data.shape)

        # Generate a validation set.
        self.num_epochs = self.config.num_epochs
        self.validation_size = int(self.train_labels.shape[0] * self.config.validation_ratio)

        self.validation_data   = self.train_data[:self.validation_size, ...]
        self.validation_labels = self.train_labels[:self.validation_size]
        self.train_data   = self.train_data[self.validation_size:, ...]
        self.train_labels = self.train_labels[self.validation_size:]

        self.train_size = self.train_labels.shape[0]
        self.test_size = self.test_labels.shape[0]
        
        self.train_index = 0  # start index of data
        self.validation_index = 0
        self.test_index = 0
    
    def next_batch(self, batch_size):
        idx = np.arange(self.train_index, self.train_index+batch_size, 1)
        idx = idx % self.train_size
        self.train_index = (self.train_index + batch_size) % self.train_size
        yield self.train_data[idx], self.train_labels[idx]
