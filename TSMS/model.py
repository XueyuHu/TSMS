import numpy as np
import pandas as pd
from numpy.random import randint
import random
import os
import warnings

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Add, Dense, Flatten, Input, Lambda, BatchNormalization, Dropout, concatenate
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from dataset import *
import argparse


class ANN_s1(Model):
    def __init__(self):
        super(ANN_s1, self).__init__()
        self.dense1 = Dense(512, activation='relu', name='dense_1', kernel_initializer='random_normal')
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(128, activation='relu', name='dense_2', kernel_initializer='random_normal')
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(0.25)
        self.dense3 = Dense(64, activation='relu', name='dense_3', kernel_initializer='random_normal')
        self.output_layer = Dense(1, name='output')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        output = self.output_layer(x)
        return output