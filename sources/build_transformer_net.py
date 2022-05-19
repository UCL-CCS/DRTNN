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

def transformer_net(inputs, head_size, num_heads, filter_dim, dropout=0.0):

    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)

    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    residual = x + inputs

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    residual2 = x + residual

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual3 = x + residual2

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual4 = x + residual3

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual5 = x + residual4

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual6 = x + residual5

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual7 = x + residual6

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual8 = x + residual7

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual9 = x + residual8

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual10 = x + residual9

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual11 = x + residual10

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual12 = x + residual11

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual13 = x + residual12

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual14 = x + residual13

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual15 = x + residual14

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual16 = x + residual15

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual17 = x + residual16

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual18 = x + residual17

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual19 = x + residual18

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual20 = x + residual19

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual21 = x + residual20

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual22 = x + residual21

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual23 = x + residual22

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual24 = x + residual23

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual25 = x + residual24
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual26 = x + residual25

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual27 = x + residual26

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual28 = x + residual27

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual29 = x + residual28

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual30 = x + residual29
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual31 = x + residual30

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual32 = x + residual31

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual33 = x + residual32

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual34 = x + residual33

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual35 = x + residual34
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual36 = x + residual35

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual37 = x + residual36

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual38 = x + residual37

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual39 = x + residual38

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual40 = x + residual39
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual41 = x + residual40

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual42 = x + residual41

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual43 = x + residual42

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual44 = x + residual43

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual45 = x + residual44
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual46 = x + residual45

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual47 = x + residual46

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual48 = x + residual47

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual49 = x + residual48

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual50 = x + residual49
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual51 = x + residual50
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual52 = x + residual51

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual53 = x + residual52

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual54 = x + residual53

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual55 = x + residual54

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual56 = x + residual55
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual57 = x + residual56

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual58 = x + residual57

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual59 = x + residual58

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual60 = x + residual59

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual61 = x + residual60

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual62 = x + residual61
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual63 = x + residual62

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual64 = x + residual63

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual65 = x + residual64

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual66 = x + residual65

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual67 = x + residual66

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual68 = x + residual67

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual69 = x + residual68

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual70 = x + residual69
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual71 = x + residual70

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual72 = x + residual71

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual73 = x + residual72

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual74 = x + residual73

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual75 = x + residual74
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual76 = x + residual75

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual77 = x + residual76

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual78 = x + residual77

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual79 = x + residual78

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual80 = x + residual79
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual81 = x + residual80
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual82 = x + residual81
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual83 = x + residual82

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual84 = x + residual83

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual85 = x + residual84

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual86 = x + residual85

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual87 = x + residual86

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual88 = x + residual87

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual89 = x + residual88

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual90 = x + residual89
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual91 = x + residual90

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual92 = x + residual91

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual93 = x + residual92

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual94 = x + residual93

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual95 = x + residual94
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual96 = x + residual95

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual97 = x + residual96

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual98 = x + residual97

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual99 = x + residual98

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual100 = x + residual99
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual101 = x + residual100
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual102 = x + residual101
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual103 = x + residual102

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual104 = x + residual103

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual105 = x + residual104

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual106 = x + residual105

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual107 = x + residual106

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual108 = x + residual107

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual109 = x + residual108

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual110 = x + residual109
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual111 = x + residual110

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual112 = x + residual111

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual113 = x + residual112

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual114 = x + residual113

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual115 = x + residual114
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual116 = x + residual115

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual117 = x + residual116

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual118 = x + residual117

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual119 = x + residual118

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual120 = x + residual119
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual121 = x + residual120

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual122 = x + residual121

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual123 = x + residual122

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual124 = x + residual123

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual125 = x + residual124
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual126 = x + residual125

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual127 = x + residual126

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual128 = x + residual127

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual129 = x + residual128

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual130 = x + residual129
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual131 = x + residual130
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual132 = x + residual131
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual133 = x + residual132

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual134 = x + residual133

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual135 = x + residual134

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual136 = x + residual135

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual137 = x + residual136

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual138 = x + residual137

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual139 = x + residual138

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual140 = x + residual139
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual141 = x + residual140

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual142 = x + residual141

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual143 = x + residual142

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual144 = x + residual143

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual145 = x + residual144
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual146 = x + residual145

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual147 = x + residual146

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual148 = x + residual147

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)
    residual149 = x + residual148

    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    residual150 = x + residual149
    x = keras.layers.LayerNormalization(epsilon=1e-6)(residual2)
    # x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=filter_dim, kernel_size=3, padding='same', activation=None)(x)
    # x = LeakyReLU(alpha=0.15)(x)
    x = keras.layers.Dropout(dropout)(x)
    # x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same', activation=None)(x)

    return x + residual150