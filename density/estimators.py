"""Concrete implementations of density estimators"""

from density.base import AbstractDensityEstimator

import numpy as np
from KDEpy import FFTKDE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import auc
from scipy.interpolate import interp1d

class GMM(AbstractDensityEstimator):
    """Wrapper around scikit GaussianMixture"""
    
    def __init__(self, n_components, covariance_type='full'):
        self._estimator = GaussianMixture(n_components, covariance_type)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    def fit(self, data):
        self._estimator.fit(data.reshape(-1,1))

    def evaluate(self, data):
        return np.exp(self._estimator.score_samples(data.reshape(-1,1)))

    def get_weights(self, data):
        return 1/self.evaluate(data)

class ConvKDE(AbstractDensityEstimator):
    """Wrapper around FFTKDE from KDEpy"""

    def __init__(self, kernel='gaussian', bw='ISJ'):
        self._estimator = FFTKDE(kernel=kernel, bw=bw)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    def fit(self, data, range=None, bins=2000):
        if range is not None:
            x_min = range[0]
            x_max = range[1]
        else:
            x_min = data.min()*0.9
            x_max = data.max()*1.1

        x_ref = np.linspace(x_min, x_max, bins)
        y = self._estimator.fit(data).evaluate(x_ref)
        y_ref = y/auc(x_ref,y)
        self.f_interp = interp1d(x_ref, y_ref, kind="cubic", assume_sorted=True)

    def evaluate(self, data):
        return self.f_interp(data)

    def get_weights(self, data):
        return 1/(self.evaluate(data))