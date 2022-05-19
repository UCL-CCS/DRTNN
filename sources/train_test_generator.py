# --------------------------------------------------
#
#     Copyright (C) {2022} Kevin Bronik
#
#     “Advancing Science Through Computers”
#     The Centre for Computational Science
#     https://www.ucl.ac.uk/computational-science/
#

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
#     This train test generator python code uses piece of a source code written by Maxime Vassaux.
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


def load_train_test_files(number_file):

    if number_file == 1:
        print('\x1b[6;30;41m' + "                        " + '\x1b[0m')
        print('\x1b[6;30;41m' + "Loading Train.csv  ...  " + '\x1b[0m')
        print('\x1b[6;30;41m' + "                        " + '\x1b[0m')
        file_path = os.path.join('./sources/Train.csv')
    else:
        print('\x1b[6;30;41m' + "                        " + '\x1b[0m')
        print('\x1b[6;30;41m' + "Loading Test.csv  ...   " + '\x1b[0m')
        print('\x1b[6;30;41m' + "                        " + '\x1b[0m')
        file_path = os.path.join('./sources/Test.csv')

    # print(file_name)

    data_pd = pd.read_csv(file_path)
    data_list = data_pd.values
    data_list = np.insert(data_list, 0, data_list[:,1]*8+data_list[:,2], axis=1)
    data_list = data_list[np.lexsort((data_list[:, 1], data_list[:, 0]))]
    T = np.max(data_list[:,1])
    nqps = np.size(np.unique(data_list[:,0]))
    return data_list, nqps, T


def generate_subdataset(data_list, nqps, T, strain_components=None, stress_components=None):
  
    subdataset = np.empty([nqps, T, len(strain_components)+len(stress_components)])
    nq = 0
    qp_ms_traj = 0
    for j in range(np.size(data_list[:,0])):
      cqid = data_list[j,0]
      time = data_list[j,1]

      features_list = []
      for component in strain_components:
        features_list.append(data_list[j,4+component])
      for component in stress_components:
        features_list.append(data_list[j,-7+component])

      if time == 1:
        qp_ms_traj = np.array([features_list])
      elif time > 1:
        qp_ms_traj = np.append(qp_ms_traj, [features_list], axis=0)
      try:
        next_cqid = data_list[j+1,0]
      except:
        next_cqid = -1

      if next_cqid != cqid:
        subdataset[nq,:,:] = qp_ms_traj
        nq+=1
    return subdataset


def generate_fulldataset_from_file_range(file_range, strain_components=[1], stress_components=[1]):

    nqps_tot = 0
    first_iter = True


    for i in file_range:
        # Reading data from the file generated by a given rank during the simulation of the finite element model
        data_list_file, nqps, T = load_train_test_files(i)
        nqps_tot += nqps
        subdataset = generate_subdataset(data_list_file, nqps, T, strain_components, stress_components)
        if first_iter:
            dataset = subdataset
            first_iter = False
        else:
            dataset = np.append(dataset, subdataset, axis=0)

    assert np.shape(dataset)[1] == (
        T), "The length of the rows of train_X ({}) does not agree with the current timestep ({})".format(
        np.shape(X)[1], T)
    assert np.shape(dataset)[
               0] == nqps_tot, "The length of the columns of train_X ({}) does not agree with the total number of qps parsed".format(
        np.shape(X)[0], nqps_tot)

    return dataset, nqps_tot, T

def normalise_dataset(dataset):
  dataset_normalised = np.empty(dataset.shape)

  normaliser = []
  for qp in np.arange(dataset.shape[0]):
    normaliser.append(MinMaxScaler())
    normaliser[qp].fit(dataset[qp,:,:])
    dataset_normalised[qp,:,:] = normaliser[qp].transform(dataset[qp,:,:])

  return dataset_normalised, normaliser


def denormalise_dataset(dataset_normalised, normaliser):
  dataset = np.empty(dataset_normalised.shape)

  for qp in np.arange(dataset_normalised.shape[0]):
    dataset[qp,:,:] = normaliser[qp].inverse_transform(dataset_normalised[qp,:,:])

  return dataset



def train_test_generator_all(arg1=None, arg2=None):
    global X, y, X_train, X_test, y_train, y_test
    file_range_data = np.arange(1, 3) 
    fulldataset, nqps_tot_data, T_data = generate_fulldataset_from_file_range(file_range_data, arg1,
                                                                              arg2)
    # print(fulldataset.shape, nqps_tot_data, T_data)
    fulldataset_wgrads = np.append(fulldataset,
                                   np.gradient(fulldataset, axis=1),
                                   axis=2)
    assert fulldataset_wgrads.shape[2] == 2 * fulldataset.shape[
        2], "The number of variables (features+targets) should have doubled after including their time differences (gradients)"
    # print(fulldataset_wgrads.shape)
    fulldataset_wgrads_wshift = np.append(fulldataset_wgrads,
                                          np.roll(fulldataset_wgrads, shift=1, axis=1),
                                          axis=2)[:, 1:, :]
    assert fulldataset_wgrads_wshift.shape[2] == 2 * fulldataset_wgrads.shape[
        2], "The number of variables (features+targets) should have doubled after including their time differences (gradients)"
    # print('fulldataset_wgrads_wshift.shape:', fulldataset_wgrads_wshift.shape)

    features_list = [4, 5, 2]
    X = fulldataset_wgrads_wshift[:, :, features_list]
    # delta_stress_t
    targets_list = [3]
    y = fulldataset_wgrads_wshift[:, :, targets_list]
    # print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_norm, X_normaliser = normalise_dataset(X)
    y_norm, y_normaliser = normalise_dataset(y)
    X_train_norm, X_train_normaliser = normalise_dataset(X_train)
    y_train_norm, y_train_normaliser = normalise_dataset(y_train)
    X_test_norm, X_test_normaliser = normalise_dataset(X_test)
    y_test_norm, y_test_normaliser = normalise_dataset(y_test)

    return X_train_norm, X_train_normaliser, y_train_norm, y_train_normaliser, X_test_norm, X_test_normaliser,y_test_norm, y_test_normaliser, X_norm, X_normaliser,y_norm, y_normaliser

