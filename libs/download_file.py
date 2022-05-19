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


def download_save_file(url, dest=None):
    """
    Download and save a file specified by url to dest directory,
    """
    u = urllib2.urlopen(url)

    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = 'downloaded.file'
    if dest:
        filename = os.path.join(dest, filename)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)
            print(status, end="")
        print()

    return filename