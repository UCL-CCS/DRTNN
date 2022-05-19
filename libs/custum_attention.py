from __future__ import (division, absolute_import, print_function, unicode_literals)
import time
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from keras import backend as K
import keras
import numpy as np
import os
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
import os
import shutil
import sys
import signal
import subprocess
import time
from keras.callbacks import EarlyStopping,  TensorBoard, LambdaCallback, ModelCheckpoint
from keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation, Lambda,Flatten,Dense,Reshape
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.advanced_activations import PReLU as prelu
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
K.set_image_data_format('channels_last')
from tensorflow.python.ops import math_ops
import math
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
import string
_CHR_IDX = string.ascii_lowercase
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
# from tensorflow.python.keras.layers import einsum_dense
# from tensorflow.python.keras.layers import core
# from tensorflow.python.framework import tensor_spec
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.framework import type_spec
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.engine import keras_tensor
# from tensorflow.python.keras.utils import tf_contextlib
# from tensorflow.python.keras.layers import advanced_activations
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import variables
# from tensorflow.python.ops.ragged import ragged_tensor
# from tensorflow.python.ops.ragged import ragged_tensor_value
# from tensorflow.python.util import nest
# from tensorflow.python.util import object_identity
#
from keras_multi_head import MultiHead

import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import packages (and setup) for plots
# import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
# assert len(tf.config.list_physical_devices('GPU')) > 0

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


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.compat.v1.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.compat.v1.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs

# def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):
#
#     T_q = tf.shape(queries)[1]
#     T_k = tf.shape(keys)[1]
#
#     Q = tf.keras.layers.Dense(queries, num_units, name='Q')
#     K_V = tf.keras.layers.Dense(keys, 2*num_units, name='K_V')
#     K, V = tf.split(K_V, 2, -1)
#
#     Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
#     K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
#     V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
#
#     align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))
#     align = align / np.sqrt(K_.get_shape().as_list()[-1])
#
#     paddings = tf.fill(tf.shape(align), float('-inf'))
#
#     key_masks = k_masks
#     key_masks = tf.tile(key_masks, [num_heads, 1])
#     key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])
#     align = tf.where(tf.equal(key_masks, 0), paddings, align)
#
#     if future_binding:
#         lower_tri = tf.ones([T_q, T_k])
#         lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
#         masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1])
#         align = tf.where(tf.equal(masks, 0), paddings, align)
#
#     align = tf.nn.softmax(align)
#     query_masks = tf.compat.v1.to_float(q_masks)
#     query_masks = tf.tile(query_masks, [num_heads, 1])
#     query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])
#     align *= query_masks
#     outputs = tf.matmul(align, V_)
#     outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
#     outputs += queries
#     outputs = layer_norm(outputs)
#     return outputs
#
# def layer_norm(inputs, epsilon=1e-8):
#     mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
#     normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
#
#     params_shape = inputs.get_shape()[-1:]
#     gamma = tf.compat.v1.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
#     beta = tf.compat.v1.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
#
#     outputs = gamma * normalized + beta
#     return outputs

def _build_attention_equation(rank, attn_axes):
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join([target_notation[i] for i in batch_dims] +
                               [target_notation[i] for i in attn_axes] +
                               [source_notation[i] for i in attn_axes])
    dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                          product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                      target_notation)
    return dot_product_equation, combine_equation, attn_scores_rank

def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)

def get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

# def multihead_attn(query, value, q_masks, k_masks, future_binding, _key_dim, _num_heads):
# def multihead_attn(query, value, _key_dim, _num_heads):
#
#     key = None
#     if key is None:
#        key = value
#     if hasattr(query, "shape"):
#         query_shape = tensor_shape.TensorShape(query.shape)
#     else:
#         query_shape = query
#     if hasattr(value, "shape"):
#         value_shape = tensor_shape.TensorShape(value.shape)
#     else:
#         value_shape = value
#     if key is None:
#         key_shape = value_shape
#     elif hasattr(key, "shape"):
#         key_shape = tensor_shape.TensorShape(key.shape)
#     else:
#         key_shape = key
#
#     free_dims = query_shape.rank - 1
#     einsum_equation, bias_axes, output_rank = _build_proj_equation(
#             free_dims, bound_dims=1, output_dims=2)
#     query = einsum_dense.EinsumDense(equation=einsum_equation,
#             output_shape=get_output_shape(output_rank - 1,
#                                            [_num_heads, _key_dim]),
#             bias_axes=None,
#             name="query")(query)
#         # einsum_equation, bias_axes, output_rank = _build_proj_equation(
#         #     key_shape.rank - 1, bound_dims=1, output_dims=2)
#     free_dims = key_shape.rank - 1
#     einsum_equation, bias_axes, output_rank = _build_proj_equation(
#           free_dims, bound_dims=1, output_dims=2)
#     key = einsum_dense.EinsumDense(equation=einsum_equation,
#                                      output_shape=get_output_shape(output_rank - 1,
#                                                                    [_num_heads, _key_dim]),
#                                      bias_axes=None,
#                                      name="key")(key)
#     free_dims = value_shape.rank - 1
#     einsum_equation, bias_axes, output_rank = _build_proj_equation(
#         free_dims, bound_dims=1, output_dims=2)
#     value = einsum_dense.EinsumDense(equation=einsum_equation,
#                                    output_shape=get_output_shape(output_rank - 1,
#                                                                  [_num_heads, _key_dim]),
#                                    bias_axes=None,
#                                    name="value")(value)
#
#
#     query = math_ops.multiply(query, 1.0 / math.sqrt(float(2.0)))
#     query_shape = tensor_shape.TensorShape(query.shape)
#     free_dims = query_shape.rank - 1
#
#     einsum_equation, bias_axes, the_rank = _build_proj_equation(free_dims, bound_dims=1, output_dims=2)
#     _dot_product_equation, _combine_equation, attn_scores_rank = _build_attention_equation(rank=the_rank, attn_axes=tuple(range(1, the_rank - 2)))
#     print('key', key)
#     print('query', query)
#     attention_scores = special_math_ops.einsum(_dot_product_equation, key, query)
#     # attention_scores = special_math_ops.einsum(_dot_product_equation, query, query)
#     _attention_axes = tuple(range(1, the_rank - 2))
#     norm_axes = tuple(
#         range(attn_scores_rank - len(_attention_axes), attn_scores_rank))
#     # attention_scores =_masked_softmax(attention_scores, attention_mask)
#     attention_scores = advanced_activations.Softmax(attention_scores, attention_mask=NameError, axis=norm_axes)
#
#     # This is actually dropping out entire tokens to attend to, which might
#     # seem a bit unusual, but is taken from the original Transformer paper.
#     attention_scores_dropout = core.Dropout(rate=attention_scores)
#     # attention_scores_dropout =_dropout_layer(
#     #     attention_scores, training=training)
#     # `context_layer` = [B, T, N, H]
#     attention_output = special_math_ops.einsum(_combine_equation,
#                                                attention_scores_dropout, value)
#     return attention_output, attention_scores
