
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
import pandas as pd

class DataGenerator:
    
    def __init__(self, config):
        self.config = config

        # Get the data, mnist.npz is in ~/.keras/datasets/mnist.npz
        print("Loading the MNIST data in ~/.keras/datasets/mnist.npz")
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = mnist.load_data()
        self.train_data   = self.train_data  .reshape(-1,28,28,1).astype(np.float32)
        self.train_labels = self.train_labels.reshape(-1)        .astype(np.int64)
        self.test_data = pd.read_csv("../test.csv").values.reshape(-1,28,28,1).astype(np.float32)
        self.test_labels = np.ones([self.test_data.shape[0]],dtype=np.int64)

        self.train_data = 2.0*self.train_data/self.config.pixel_depth - 1.0
        self.test_data  = 2.0*self.test_data /self.config.pixel_depth - 1.0

        print("train data shape:", self.train_data.shape)
        print("test  data shape:", self.test_data.shape)

        # Generate a validation set.
        self.num_epochs = self.config.num_epochs
        self.valid_size = int(self.train_labels.shape[0] * self.config.validation_ratio)

        self.valid_data   = self.train_data[:self.valid_size, ...]
        self.valid_labels = self.train_labels[:self.valid_size]
        # self.train_data   = self.train_data[self.valid_size:, ...]
        # self.train_labels = self.train_labels[self.valid_size:]

        self.train_size = self.train_labels.shape[0]
        self.test_size = self.test_labels.shape[0]
        
        self.train_index = 0  # start index of data
        self.valid_index = 0
        self.test_index = 0
    
    def train_next_batch(self, batch_size):
        idx = np.arange(self.train_index, self.train_index+batch_size, 1)
        idx = idx % self.train_size
        self.train_index = (self.train_index + batch_size) % self.train_size
        yield self.train_data[idx], self.train_labels[idx]
    
    def valid_next_batch(self, batch_size):
        idx = np.arange(self.valid_index, self.valid_index+batch_size, 1)
        idx = idx % self.valid_size
        self.valid_index = (self.valid_index + batch_size) % self.valid_size
        yield self.valid_data[idx], self.valid_labels[idx]

    def test_next_batch(self, batch_size):
        idx = np.arange(self.test_index, self.test_index+batch_size, 1)
        idx = idx % self.test_size
        self.test_index = (self.test_index + batch_size) % self.test_size
        yield self.test_data[idx], self.test_labels[idx]

