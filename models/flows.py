"""Module of Normalizing Flow implementations in Keras"""

from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
tfkl = tfk.layers

class Flow(tfk.Model, ABC):
    """Base abstract class of a Normalizing Flow model.
    
    This abstraction inherits from the 'Model' class

    Suclasses inheriting from 'Flow' should implement the methods below:
      'call': Returns the 'log_prob' of the inputs.
      'forward': Transforms the input using the forward bijection.
      'inverse': Transforms the input using the inverse bijection.
      'sample': Generates a data sampled from the model
    """
    def __init__(self, **kwargs):
        super(Flow, self).__init__(**kwargs)

    @abstractmethod
    def call(self, inputs):
        pass

    @abstractmethod
    def forward(self, x): 
        pass

    @abstractmethod
    def inverse(self, y): 
        pass

    @abstractmethod
    def sample(self, sample_shape): 
        pass


class MAF(Flow):
    """Masked Autoregressive Flow model.
    
    Based a chain of trainable 'Bijector' objects from the 
    'tensorflow-probability' module
    """
    def __init__(self, n_dims, n_layers, units, name=None, dtype='float32'):
        """Creates a MAF object.

        Args:
          n_dims: The dimensionality of input (and output) data
          n_layers: The number of 'AutoregressiveNetwork' bijectors
          units: The units of every 'AutoregressiveNetwork' object
          name: String argument passed to the 'Model' parent class constructor
          dtype: The type of data used by the network
        """ 
        super(MAF, self).__init__(name=name, dtype=dtype)

        # Create the necessary 'AutoregressiveNetwork' bijectors
        self.mades = [tfb.AutoregressiveNetwork(
                            params=2, 
                            hidden_units=units,
                            input_order=self._input_order(i, n_layers),
                            activation='tanh')
                     for i 
                     in range(n_layers)]

        """ Make a chain of MAF bijectors passing the previously defined
        MADEs as the 'shift_and_log_scale_fn' """
        self.bijectors = [tfb.MaskedAutoregressiveFlow(made)
                          for made 
                          in self.mades]
        self.chain = tfb.Chain(self.bijectors)

        """Use bijector chain to make a TransformedDistribution starting 
        from a Normal base distribution.""" 
        self.distribution = tfd.TransformedDistribution(
                tfd.Normal(loc=0., scale=1.),
                bijector=self.chain,
                event_shape=[n_dims])

    def call(self, x):
        """Returns the log likelihood of the input samples"""
        return self.distribution.log_prob(x)

    def forward(self, x):
        """Returns the input transformed by the forward bijector chain."""
        return self.chain(x).numpy()

    def inverse(self, y):
        """Returns the input transformed by the inverse bijector chain."""
        return self.chain.inverse(y).numpy()

    def sample(self, sample_shape=()):
        """Returns a sample from the transformed distribution"""
        return self.distribution.sample(sample_shape).numpy()

    def forward_log_det_jacobian(self, x, event_ndims=1):
        """Log of the Jacobian determinant for a forwad transofrmation."""
        return self.chain.forward_log_det_jacobian(x, event_ndims=1).numpy()

    def inverse_log_det_jacobian(self, x, event_ndims=1):
        """Log of the Jacobian determinant for an inverse transofrmation."""
        return self.chain.inverse_log_det_jacobian(x, event_ndims=1).numpy()

    def _input_order(self, i, n_layers):
        """Determine input order of a Layer given its index number"""
        order = 'random'
        if i == 0:
            order = 'left-to-right'
        elif i == n_layers-1:
            order = 'right-to-left'
        return order