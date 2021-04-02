""" Module of keras autoencoders created with subclassing"""

from functools import reduce
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np

class DenseEncoder(tfkl.Layer):
    """Fully connected encoder made of 'Dense' layers."""
    def __init__(self,
                 encoding_dim,
                 units_list,
                 hidden_activation,
                 output_activation,
                 weight_reg=None,
                 bias_reg=None,
                 output_reg=None,
                 dtype=None,
                 name=None):
        """Crates a DenseEncoder object.

        Args:
          encoding_dim: The number of nodes in the encoder output.
          unitsl_list: List of integers. For every list element a Dense layer
             will be crated and the number of nodes will be set to the value 
             of the element
          hidden_activation: The activation funtion used for all the layers 
            apart from the output layer.
          output_activation: The activation functiun used for the output layer
          weight_reg: 'kernel_regularizer' of all layers.
          bias_reg: 'bias_regularizer' of all layers.
          output_reg: 'activity_regularizer' for all layers.
          dtype: Data type, this argument is passed to the parent 'Layer' 
            class constructor
          name: Object name, string to be passed to the parent class 
            constructor
        """
        super(DenseEncoder, self).__init__(name=name, dtype=dtype)
        # Create the list of the hidden layers
        self.hidden_layers = [
            tfkl.Dense(units, activation=hidden_activation, dtype=dtype,
                       kernel_regularizer=weight_reg,
                       bias_regularizer=bias_reg,
                       activity_regularizer=output_reg) 
            for units 
            in units_list]
        
        # Creates the output layer
        self.output_layer = tfkl.Dense(encoding_dim, dtype=dtype,
                                       activation=output_activation,
                                       kernel_regularizer=weight_reg,
                                       bias_regularizer=bias_reg,
                                       activity_regularizer=output_reg)

    def call(self, inputs):
        """Runs the inputs through all of encoder layers"""
        activation = reduce(lambda x, layer: layer(x), self.hidden_layers, 
                            inputs)
        return self.output_layer(activation)


class DenseDecoder(tfkl.Layer):
    """Fully connected decoders made of 'Dense' layers."""
    def __init__(self,
                 output_dim,
                 units_list,
                 hidden_activation,
                 output_activation,
                 weight_reg=None,
                 bias_reg=None,
                 output_reg=None,
                 dtype=None,
                 name=None):
        """Crates a DenseDecoder object.

        Args:
          encoding_dim: The number of nodes in the decoder output.
          unitsl_list: List of integers. For every list element a Dense layer
             will be crated and the number of nodes will be set to the value 
             of the element
          hidden_activation: The activation funtion used for all the layers 
            apart from the output layer.
          output_activation: The activation functiun used for the output layer
          weight_reg: 'kernel_regularizer' of all layers.
          bias_reg: 'bias_regularizer' of all layers.
          output_reg: 'activity_regularizer' for all layers.
          dtype: Data type, this argument is passed to the parent 'Layer' 
            class constructor
          name: Object name, string to be passed to the parent class 
            constructor
        """
        super(DenseDecoder, self).__init__(name=name, dtype=dtype)
        # Creates the list of hidden layers 
        self.hidden_layers = [
            tfkl.Dense(units, activation=hidden_activation, dtype=dtype,
                       kernel_regularizer=weight_reg,
                       bias_regularizer=bias_reg,
                       activity_regularizer=output_reg) 
            for units 
            in units_list] 

        # Creates the output layer
        self.output_layer = tfkl.Dense(output_dim, dtype=dtype,
                                       activation=output_activation,
                                       kernel_regularizer=weight_reg,
                                       bias_regularizer=bias_reg,
                                       activity_regularizer=output_reg)

    def call(self, inputs):
        """Run the inputs through all of the decoder layers."""
        activation = reduce(lambda x, layer: layer(x), self.hidden_layers, 
                            inputs)
        return self.output_layer(activation)

class Autoencoder(tfk.Model, ABC):
    """Base abstract class of an Autoencoder.
    
    This autoencoder abstraction inherits from the 'Model' class

    Suclasses inheriting from 'Autoencoder' should implement the methods for:
      'call': Returns the autoencoder reconstruction of the inputs
      'encode': Encodes the inputs.
      'decode': Decodes a given lattent representation.
    """
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

    @abstractmethod
    def call(self, inputs):
        pass

    @abstractmethod
    def encode(self, inputs):
        pass

    @abstractmethod
    def decode(self, outputs):
        pass

class DenseAutoencoder(Autoencoder):
    """Fully connected autoencoder network model.
    
    Contains one instance of 'DenseEncoder' and one instance 'DenseDecoder'.
    """
    def __init__(self, 
                 input_dim, 
                 encoding_dim, 
                 units=[],
                 hidden_activation=tf.nn.relu,
                 output_activation=tf.nn.sigmoid,
                 bias_reg=None,
                 weight_reg=None,
                 output_reg=None,
                 dtype='float32',
                 name=None):
        """Creates a 'DenseAutoencoder' object.
        
        Args:
          input_dim: The number of nodes for the autoencoder input and output
          encoding_dim: The number of nodes in the encoder output.
          units: List of integers. This list is passed to the 
            'DenseEncoder' constructor. The same list but reversed is passed
            to the 'DenseDecoder' constructor
          hidden_activation: The activation funtion used for all the layers 
            apart from the encoder and decoder output layers.
          output_activation: The activation functiun used for the output layer
            of bouth the encoder and the decoder.
          weight_reg: 'kernel_regularizer' of all layers.
          bias_reg: 'bias_regularizer' of all layers.
          output_reg: 'activity_regularizer' for all layers.
          dtype: Data type, this argument is passed to the parent 'Model' 
            class constructor
          name: Object name, string to be passed to the parent class 
            constructor
        """ 
        super(DenseAutoencoder, self).__init__(name=name, dtype=dtype)
        self.encoder = DenseEncoder(encoding_dim, units, 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               dtype=dtype)
        self.decoder = DenseDecoder(input_dim, units[::-1], 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               dtype=dtype)

        self.encoder(np.zeros(shape=(1,input_dim)))
    
    def call(self, inputs):
        """Run the inputs through the full autoencoder"""
        encoding = self.encoder(inputs)
        reconstructed = self.decoder(encoding)
        return reconstructed
    
    def encode(self, inputs):
        """Genereate the latent representation of the inputs"""
        return self.encoder(inputs).numpy()

    def decode(self, encoding):
        """Reconstruct the inputs using a given latent representation"""
        return self.decoder(encoding).numpy()

