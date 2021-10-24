"""Defines the base abstractions for building DataLoaders."""

from abc import ABC, abstractmethod, abstractproperty

import numpy as np

class AbstractDensityEstimator(ABC):
    """Abstract base class of a density estimator.

    Any Desnisty estimator must implement the methods 'fit',
    'evaluate' and 'get_weights'
    """
    @abstractproperty
    def estimator(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def scale_down(self, x):
        return x/np.min(x)