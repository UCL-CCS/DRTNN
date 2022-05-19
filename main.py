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
from shutil import rmtree
K.set_image_data_format('channels_last')

import string
_CHR_IDX = string.ascii_lowercase
import signal
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

from sources.train_test_generator import train_test_generator_all, denormalise_dataset, normalise_dataset
# from DTNN import dtnn_init
from sources import DRTNN_model

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


batch_size = 1
def run_train_test(model, model_weight_load=False):

    print('\x1b[6;30;41m' + "                               " + '\x1b[0m')
    print('\x1b[6;30;41m' + "Loading data into memory  ...  " + '\x1b[0m')
    print('\x1b[6;30;41m' + "                               " + '\x1b[0m')

    strain_component_epsilon11 = [4]  # σ11, ε11
    stress_component_sigma11 = [4]  # σ11, ε11
    strain_component_epsilon22 = [6]  # σ22, ε22
    stress_component_sigma22 = [6]  # σ22, ε22
    strain_component_epsilon33 = [1]  # σ33, ε33
    stress_component_sigma33 = [1]  # σ33, ε33

    X_train_norm11, X_train_normaliser11, y_train_norm11, y_train_normaliser11, X_test_norm11, X_test_normaliser11, y_test_norm11, y_test_normaliser11, X_norm11, X_normaliser11, y_norm11, y_normaliser11 = train_test_generator_all(
        arg1=strain_component_epsilon11, arg2=stress_component_sigma11)
    X_train_norm22, X_train_normaliser22, y_train_norm22, y_train_normaliser22, X_test_norm22, X_test_normaliser22, y_test_norm22, y_test_normaliser22, X_norm22, X_normaliser22, y_norm22, y_normaliser22 = train_test_generator_all(
        arg1=strain_component_epsilon22, arg2=stress_component_sigma22)
    X_train_norm33, X_train_normaliser33, y_train_norm33, y_train_normaliser33, X_test_norm33, X_test_normaliser33, y_test_norm33, y_test_normaliser33, X_norm33, X_normaliser33, y_norm33, y_normaliser33 = train_test_generator_all(
        arg1=strain_component_epsilon33, arg2=stress_component_sigma33)

    strain_component_epsilon01 = [2]  # σ01, ε01
    stress_component_sigma01 = [2]  # σ01, ε01
    strain_component_epsilon02 = [3]  # σ02, ε02
    stress_component_sigma02 = [3]  # σ02, ε02
    strain_component_epsilon12 = [5]  # σ12, ε12
    stress_component_sigma12 = [5]  # σ12, ε12

    X_train_norm01, X_train_normaliser01, y_train_norm01, y_train_normaliser01, X_test_norm01, X_test_normaliser01, y_test_norm01, y_test_normaliser01, X_norm01, X_normaliser01, y_norm01, y_normaliser01 = train_test_generator_all(
        arg1=strain_component_epsilon01, arg2=stress_component_sigma01)
    X_train_norm02, X_train_normaliser02, y_train_norm02, y_train_normaliser02, X_test_norm02, X_test_normaliser02, y_test_norm02, y_test_normaliser02, X_norm02, X_normaliser02, y_norm02, y_normaliser02 = train_test_generator_all(
        arg1=strain_component_epsilon02, arg2=stress_component_sigma02)
    X_train_norm12, X_train_normaliser12, y_train_norm12, y_train_normaliser12, X_test_norm12, X_test_normaliser12, y_test_norm12, y_test_normaliser12, X_norm12, X_normaliser12, y_norm12, y_normaliser12 = train_test_generator_all(
        arg1=strain_component_epsilon12, arg2=stress_component_sigma12)

    def sigma_epsilon_one_two_three_four_five_six(model, epoch, comp=None, Xtest11=None, Xtest22=None, Xtest33=None,
                                                  Xtest01=None, Xtest02=None, Xtest12=None, Ytest=None, norml=None):
        Image_save = T_PATH + '/image_outputs_{}/img_epoch_{}.png'
        ys_predict = model.predict([Xtest11, Xtest22, Xtest33, Xtest01, Xtest02, Xtest12])

        if comp == 'σ11_ε11':
            y_predict = ys_predict[0]
        if comp == 'σ22_ε22':
            y_predict = ys_predict[1]
        if comp == 'σ33_ε33':
            y_predict = ys_predict[2]
        if comp == 'σ01_ε01':
            y_predict = ys_predict[3]
        if comp == 'σ02_ε02':
            y_predict = ys_predict[4]
        if comp == 'σ12_ε12':
            y_predict = ys_predict[5]

        y_pred_out = denormalise_dataset(y_predict, norml)
        y_test_out = denormalise_dataset(Ytest, norml)
        fig = plt.figure(figsize=(9, 6), dpi=80)
        ax = plt.axes(xlabel='timestep', ylabel='stress increment (Pa)',
                      title="Evolutions of delta_stress")

        plt.plot(y_pred_out[0, :, 0].T, label='prediction')
        plt.plot(y_test_out[0, :, 0].T, linestyle=':', label='testing data')
        plt.legend()
        plt.savefig(Image_save.format(comp, epoch))

        plt.close()

    def plot_callback_full_stress_strain(model, epoch):

        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ11_ε11', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm11, norml=y_test_normaliser11)
        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ22_ε22', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm22, norml=y_test_normaliser22)
        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ33_ε33', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm33, norml=y_test_normaliser33)

        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ01_ε01', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm01, norml=y_test_normaliser01)
        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ02_ε02', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm02, norml=y_test_normaliser02)
        sigma_epsilon_one_two_three_four_five_six(model, epoch, comp='σ12_ε12', Xtest11=X_test_norm11,
                                                  Xtest22=X_test_norm22,
                                                  Xtest33=X_test_norm33, Xtest01=X_test_norm01, Xtest02=X_test_norm02,
                                                  Xtest12=X_test_norm12, Ytest=y_test_norm12, norml=y_test_normaliser12)

    try:
        print("Model summary ...")
        model.summary()

        print('\x1b[6;30;41m' + "                                 " + '\x1b[0m')
        print('\x1b[6;30;41m' + "Loading data into memory is done!" + '\x1b[0m')
        print('\x1b[6;30;41m' + "                                 " + '\x1b[0m')

        print("Model compile ...")

        model.compile(
            loss=['mse', 'mse', 'mse', 'mse', 'mse', 'mse'],
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            # metrics=['mae', 'mape'])
            metrics=['mae'])

        # print("Save Model...")
        # from keras.utils.vis_utils import plot_model
        # tf.keras.utils.plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=300, verbose=1,
            mode='auto', min_delta=0.001, cooldown=0, min_lr=0)

        work_dir_model = os.path.join(
            T_PATH,
            'MODEL')

        model_weights_load = T_PATH + '/MODEL/' + 'model_weights_transformer.hdf5'
        if os.path.exists(
                os.path.join(model_weights_load)) and model_weight_load is True:
            n_weights = os.path.join(model_weights_load)
            model.load_weights(n_weights, by_name=True)
            print('loading weight...')
        else:
            if os.path.exists(work_dir_model):
                rmtree(work_dir_model)
            os.mkdir(work_dir_model)


        print("Training begins ...")
        model.fit(
            [X_train_norm11, X_train_norm22, X_train_norm33, X_train_norm01, X_train_norm02, X_train_norm12],
            [y_train_norm11, y_train_norm22, y_train_norm33, y_train_norm01, y_train_norm02, y_train_norm12],
            # validation_data=(X_test_norm, y_test_norm),
            batch_size=batch_size,
            validation_split=0.2,
            epochs=1400,
            verbose=1,
            callbacks=[reduce_lr, ModelCheckpoint(model_weights,
                                                  monitor='val_loss',
                                                  save_best_only=True,
                                                  save_weights_only=True),
                       EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=600,
                                     verbose=0,
                                     mode='auto'),
                       TensorBoard(log_dir=tensorboardlogs, histogram_freq=0,
                                   write_graph=True, write_images=True),
                       LambdaCallback(
                           on_epoch_end=lambda epoch, logs: plot_callback_full_stress_strain(model, epoch)
                       )
                       ])

    except:
        print("system/file error, terminating!")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)



if __name__ == "__main__":
    T_PATH = os.path.split(os.path.realpath(__file__))[0]
    # work_dir_model = os.path.join(
    #     T_PATH,
    #     'MODEL')
    work_dir_tensorboardlog = os.path.join(
        T_PATH,
        'Tensorboardlogs')

    # if os.path.exists(work_dir_model):
    #     rmtree(work_dir_model)
    # os.mkdir(work_dir_model)

    if os.path.exists(work_dir_tensorboardlog):
        rmtree(work_dir_tensorboardlog)
    os.mkdir(work_dir_tensorboardlog)
    comps = ['σ11_ε11', 'σ33_ε33', 'σ22_ε22', 'σ01_ε01', 'σ02_ε02', 'σ12_ε12']

    for comp in comps:

        ct = os.path.join(
            T_PATH,
            'image_outputs_{}'.format(comp))
        if os.path.exists(ct):
            rmtree(ct)
        os.mkdir(ct)

    sys.path.append(os.path.join(T_PATH, 'MODEL'))
    model_weights = T_PATH + '/MODEL/' + 'model_weights_transformer.hdf5'
    tensorboardlogs = T_PATH + '/Tensorboardlogs'
    input_shape = 3
    head_size = 2048
    num_heads = 4
    ff_dim = 4
    num_transformer_blocks = 4
    dropout = 0.0001
    object = DRTNN_model.DRTNN(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, dropout)
    run_train_test(model=object.model, model_weight_load=False)
