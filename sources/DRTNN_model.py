# --------------------------------------------------
#
#     Copyright (C) {2022} Kevin Bronik
#
#     “Advancing Science Through Computers”
#     The Centre for Computational Science
#     https://www.ucl.ac.uk/computational-science/


#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     [Deep Residual Transformer Neural Network (DRTNN)]
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.

from __future__ import (division, absolute_import, print_function, unicode_literals)
import time
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from keras import backend as K
import keras
import numpy as np
from keras.callbacks import EarlyStopping,  TensorBoard, LambdaCallback, ModelCheckpoint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

K.set_image_data_format('channels_last')

import string
_CHR_IDX = string.ascii_lowercase

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

plt.style.use('dark_background')
import sys, os, tempfile, logging

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

from sources.build_transformer_net import transformer_net

CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'



class DRTNN:
    """
    Class for deep residual transformer neural network.
    Name:  DRTNN

    [Deep Residual Transformer Neural Network (DRTNN)]
    """

    def __init__(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dropout, **kwargs):
        """
        Initialize the neural transformer network.

        """
        self.input_shape = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.init_network(**kwargs)
        self.print_network_info()

    def init_network(self, **kwargs):

        print('\x1b[6;30;41m' + "                            " + '\x1b[0m')
        print('\x1b[6;30;41m' + "Initiation of network  ...  " + '\x1b[0m')
        print('\x1b[6;30;41m' + "                            " + '\x1b[0m')

        in1 = keras.layers.Input(shape=(None, self.input_shape), name='in1')
        in2 = keras.layers.Input(shape=(None, self.input_shape), name='in2')
        in3 = keras.layers.Input(shape=(None, self.input_shape), name='in3')
        in4 = keras.layers.Input(shape=(None, self.input_shape), name='in4')
        in5 = keras.layers.Input(shape=(None, self.input_shape), name='in5')
        in6 = keras.layers.Input(shape=(None, self.input_shape), name='in6')
        # in2 = keras.layers.Input(shape=(None, 17), name='in2',  ragged=False)
        merged = keras.layers.concatenate([in1, in2, in3, in4, in5, in6], axis=-1)
        # print('merged', merged)
        x = merged

        x1 = x
        x2 = x
        x3 = x
        x4 = x
        x5 = x
        x6 = x
        # 1024  --->  4 to 6
        for _ in range(self.num_transformer_blocks):
            x1 = transformer_net(x1, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)
            x2 = transformer_net(x2, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)
            x3 = transformer_net(x3, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)
            x4 = transformer_net(x4, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)
            x5 = transformer_net(x5, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)
            x6 = transformer_net(x6, head_size=self.head_size, num_heads=self.num_heads, filter_dim=self.ff_dim,
                                 dropout=self.dropout)

        outputs1 = keras.layers.Dense(1)(x1)
        outputs2 = keras.layers.Dense(1)(x2)
        outputs3 = keras.layers.Dense(1)(x3)
        outputs4 = keras.layers.Dense(1)(x4)
        outputs5 = keras.layers.Dense(1)(x5)
        outputs6 = keras.layers.Dense(1)(x6)
        self.model = keras.Model(inputs=[in1, in2, in3, in4, in5, in6],
                                 outputs=[outputs1, outputs2, outputs3, outputs4, outputs5, outputs6])

    def print_network_info(self):
        """
        Print some characteristics of the neural network.
        """
        print('\x1b[6;30;41m' + "Neural net parameters:" + '\x1b[0m')
        print('Head size =', self.head_size)
        print('Number of heads =', self.num_heads)
        print('Dimension of filters =', self.ff_dim)
        print('Number of transformer blocks=', self.num_transformer_blocks)
        print('Number of dropout=', self.dropout)
        print('\x1b[6;30;41m' + "----------------------" + '\x1b[0m')


