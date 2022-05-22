""" Module of keras autoencoders created with subclassing"""

from functools import reduce
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np

class ConvBlock2D(tfkl.Layer):
    def __init__(self,
                 filters,
                 hidden_activation,
                 maxpool=2,
                 padding="same",
                 kernel_size=3,
                 weight_reg=None,
                 bias_reg=None,
                 output_reg=None,
                 dtype=None,
                 name=None):
        super(ConvBlock2D, self).__init__(name=name, dtype=dtype)
        # Create the list of the hidden layers
        self.layers = [
            tfkl.Conv2D(filters, kernel_size, padding=padding, 
                       activation=hidden_activation, dtype=dtype,
                       kernel_regularizer=weight_reg,
                       bias_regularizer=bias_reg,
                       activity_regularizer=output_reg)
        ]
        if maxpool is not None:
              self.layers.append(
            tfkl.MaxPool2D(maxpool)
        )

    def call(self, inputs):
        """Runs the inputs through the block's layers"""
        return reduce(lambda x, layer: layer(x), self.layers, 
                            inputs)

class ConvBlock2DTranspose(tfkl.Layer):
    def __init__(self,
                 filters,
                 hidden_activation,
                 maxpool=2,
                 padding="same",
                 kernel_size=3,
                 weight_reg=None,
                 bias_reg=None,
                 output_reg=None,
                 dtype=None,
                 name=None):
        super(ConvBlock2DTranspose, self).__init__(name=name, dtype=dtype)
        # Create the list of the hidden layers
       
        self.layers = [
                tfkl.UpSampling2D(maxpool)
              ]  if maxpool is not None else []
        self.layers.append(
            tfkl.Conv2DTranspose(filters, kernel_size, padding=padding, 
                       activation=hidden_activation, dtype=dtype,
                       kernel_regularizer=weight_reg,
                       bias_regularizer=bias_reg,
                       activity_regularizer=output_reg)
        )

    def call(self, inputs):
        """Runs the inputs through the block's layers"""
        return reduce(lambda x, layer: layer(x), self.layers, 
                            inputs)
         

class DenseTied(tfkl.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = tfk.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tfk.initializers.get(kernel_initializer)
        self.bias_initializer = tfk.initializers.get(bias_initializer)
        self.kernel_regularizer = tfk.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tfk.regularizers.get(bias_regularizer)
        self.activity_regularizer = tfk.regularizers.get(activity_regularizer)
        self.kernel_constraint = tfk.constraints.get(kernel_constraint)
        self.bias_constraint = tfk.constraints.get(bias_constraint)
        self.input_spec = tfkl.InputSpec(min_ndim=2)
        self.supports_masking = True
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        #self.input_spec = tfkl.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format='N..C')
        if self.activation is not None:
            output = self.activation(output)
        return output

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
                 encoder_tied=None,
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
        if not encoder_tied:
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
        else:
          # Creates the list of hidden layers 
          n_layers = len(units_list)
          self.hidden_layers = [
              DenseTied(units, activation=hidden_activation, dtype=dtype,
                        kernel_regularizer=weight_reg,
                        bias_regularizer=bias_reg,
                        activity_regularizer=output_reg,
                        tied_to=encoder_tied.hidden_layers[n_layers-i-1]
                                if i != n_layers-1 else encoder_tied.output_layer) 
              for i, units 
              in enumerate(units_list)] 

          # Creates the output layer
          self.output_layer = DenseTied(output_dim, dtype=dtype,
                                        activation=output_activation,
                                        kernel_regularizer=weight_reg,
                                        bias_regularizer=bias_reg,
                                        activity_regularizer=output_reg,
                                        tied_to=encoder_tied.hidden_layers[0])
    def call(self, inputs):
        """Run the inputs through all of the decoder layers."""
        activation = reduce(lambda x, layer: layer(x), self.hidden_layers, 
                            inputs)
        return self.output_layer(activation)

class ConvEncoder(tfkl.Layer):
    """Fully connected encoder made of 'Dense' layers."""
    def __init__(self,
                 encoding_dim,
                 filters_list,
                 hidden_activation,
                 output_activation,
                 maxpool=2,
                 padding="same",
                 kernel_size=3,
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
        super(ConvEncoder, self).__init__(name=name, dtype=dtype)
        # Create the list of the hidden layers
        self.hidden_layers = [
            ConvBlock2D(filters, hidden_activation, kernel_size=kernel_size, 
                       padding=padding, dtype=dtype, maxpool=maxpool,
                       weight_reg=weight_reg,
                       bias_reg=bias_reg,
                       output_reg=output_reg) 
            for filters 
            in filters_list]
        
        # Creates the output layer
        self.hidden_layers.append(tfkl.Flatten())
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


class ConvDecoder(tfkl.Layer):
    """Fully connected decoders made of 'Dense' layers."""
    def __init__(self,
                 n_chanels,
                 dense_dim,
                 filters_list,
                 hidden_activation,
                 input_activation,
                 maxpool=2,
                 padding="same",
                 kernel_size=3,
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
        super(ConvDecoder, self).__init__(name=name, dtype=dtype)
          # Creates the list of hidden layers 
        dense_root = int(np.sqrt(dense_dim))
        self.layers = [
          tfkl.Dense(dense_dim, dtype=dtype,
                    activation=input_activation,
                    kernel_regularizer=weight_reg,
                    bias_regularizer=bias_reg,
                    activity_regularizer=output_reg)
        ]
        self.layers += [tfkl.Reshape((dense_root, dense_root, 1))]
        self.layers += [
            ConvBlock2DTranspose(filters, hidden_activation, 
                       kernel_size=kernel_size, 
                       padding=padding, dtype=dtype, maxpool=maxpool,
                       weight_reg=weight_reg,
                       bias_reg=bias_reg,
                       output_reg=output_reg) 
            for filters 
            in filters_list]
        self.layers += [
            tfkl.Conv2DTranspose(n_chanels, kernel_size, padding=padding, 
                       activation=hidden_activation, dtype=dtype,
                       kernel_regularizer=weight_reg,
                       bias_regularizer=bias_reg,
                       activity_regularizer=output_reg)]
      
    def call(self, inputs):
        """Run the inputs through all of the decoder layers."""
        return reduce(lambda x, layer: layer(x), self.layers, 
                            inputs)

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
    def __init__(self, input_dim, encoding_dim, 
                  units=None, n_layers=None, hidden_activation=tf.nn.relu, 
                  output_activation=tf.nn.sigmoid, bias_reg=None, 
                  weight_reg=None, output_reg=None, dtype='float32', 
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
        if units is None:
            units = []
        super(DenseAutoencoder, self).__init__(name=name, dtype=dtype)
        if not units:
          if not n_layers:
            raise ValueError("Either 'units' or 'n_layers' must be provided")
          units = np.linspace(input_dim, encoding_dim, 2+n_layers, 
                              dtype=int)[1:-1]
        self.encoder = DenseEncoder(encoding_dim, units, 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype)
        self.decoder = DenseDecoder(input_dim, units[::-1], 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype)

        self.encoder(np.zeros(shape=(1,input_dim)))
    
    def call(self, inputs):
        """Run the inputs through the full autoencoder"""
        encoding = self.encoder(inputs)
        return self.decoder(encoding)
    
    def encode(self, inputs):
        """Genereate the latent representation of the inputs"""
        return self.encoder(inputs).numpy()

    def decode(self, encoding):
        """Reconstruct the inputs using a given latent representation"""
        return self.decoder(encoding).numpy()


class DenseAutoencoderTied(Autoencoder):
    """Fully connected autoencoder network model.
    
    Contains one instance of 'DenseEncoder' and one instance 'DenseDecoder'.
    """
    def __init__(self, input_dim, encoding_dim, units=None, n_layers=None, 
                  hidden_activation=tf.nn.relu, 
                  output_activation=tf.nn.sigmoid, bias_reg=None, 
                  weight_reg=None, output_reg=None, dtype='float32', 
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
        if units is None:
            units = []
        super(DenseAutoencoderTied, self).__init__(name=name, dtype=dtype)
        if not units:
          if not n_layers:
            raise ValueError("Either 'units' or 'n_layers' must be provided")
          units = np.linspace(input_dim, encoding_dim, 2+n_layers, 
                              dtype=int)[1:-1]
        self.encoder = DenseEncoder(encoding_dim, units, 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype)
        self.decoder = DenseDecoder(input_dim, units[::-1], 
                               hidden_activation, output_activation,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype,
                               encoder_tied=self.encoder)

        self.encoder(np.zeros(shape=(1,input_dim)))
    
    def call(self, inputs):
        """Run the inputs through the full autoencoder"""
        encoding = self.encoder(inputs)
        return self.decoder(encoding)
    
    def encode(self, inputs):
        """Genereate the latent representation of the inputs"""
        return self.encoder(inputs).numpy()

    def decode(self, encoding):
        """Reconstruct the inputs using a given latent representation"""
        return self.decoder(encoding).numpy()

class ConvAutoencoder(Autoencoder):
    """Fully connected autoencoder network model.
    
    Contains one instance of 'DenseEncoder' and one instance 'DenseDecoder'.
    """
    def __init__(self, 
                 input_dim, 
                 encoding_dim, 
                 units=None,
                 maxpool=2,
                 n_layers=None,
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
        if not units:
          if not n_layers:
            raise ValueError("Either 'units' or 'n_layers' must be provided")
          units = np.linspace(input_dim, encoding_dim, 2+n_layers, 
                              dtype=int)[1:-1]
        self.encoder = DenseEncoder(encoding_dim, units, 
                               hidden_activation, output_activation,
                               maxpool=2,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype)
        self.decoder = DenseDecoder(input_dim, units[::-1], 
                               hidden_activation, output_activation,
                               maxpool=2,
                               bias_reg=bias_reg, weight_reg=weight_reg,
                               output_reg=output_reg, dtype=dtype)

        self.encoder(np.zeros(shape=(1,input_dim)))
    
    def call(self, inputs):
        """Run the inputs through the full autoencoder"""
        encoding = self.encoder(inputs)
        return self.decoder(encoding)
    
    def encode(self, inputs):
        """Genereate the latent representation of the inputs"""
        return self.encoder(inputs).numpy()

    def decode(self, encoding):
        """Reconstruct the inputs using a given latent representation"""
        return self.decoder(encoding).numpy()