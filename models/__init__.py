"""Module of tensorflow models used by the PAE"""

import tensorflow.keras.optimizers as tfko
import tensorflow.nn as tfnn
import tensorflow.keras.activations as tfka
import tensorflow.keras.regularizers as tfkr
import tensorflow.keras.callbacks as tfc

from models.autoencoder import DenseAutoencoder
from models.flows import MAF

OPTIMIZERS = {
    'adadelta': tfko.Adadelta,
    'adagrad': tfko.Adagrad,
    'adam': tfko.Adam,
    'adamax': tfko.Adamax,
    'ftrl': tfko.Ftrl,
    'nadam': tfko.Nadam,
    'RMSprop': tfko.RMSprop,
    'SGD': tfko.SGD
    }

ACTIVATIONS = {
    'crelu': tfnn.crelu,
    'elu': tfnn.elu,
    'exp': tfka.exponential,
    'gelu': tfnn.gelu,
    'hard_sigmoid': tfka.hard_sigmoid,
    'leaky_relu': tfnn.leaky_relu,
    'linear': tfka.linear,
    'log_softmax': tfnn.log_softmax,
    'relu': tfnn.relu,
    'relu6': tfnn.relu6,
    'selu': tfnn.selu,
    'sigmoid': tfnn.sigmoid,
    'silu': tfnn.silu,
    'softmax': tfnn.softmax,
    'softplus': tfnn.softplus,
    'softsign': tfnn.softsign,
    'swish': tfnn.swish,
    'tanh': tfnn.tanh,
    }

REGULARIZERS = {
    'l1': tfkr.L1,
    'l2': tfkr.L2,
    'l1_l2': tfkr.L1L2
    }

MODELS = {
    'dense_ae': DenseAutoencoder,
    'maf': MAF
    }

CALLBACKS = {
    'reduce_lr_on_plateau': tfc.ReduceLROnPlateau
    }