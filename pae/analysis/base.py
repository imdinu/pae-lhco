"""Base class for a PAE analysis """

from abc import ABC, abstractproperty, abstractmethod

from density.estimators import GMM, ConvKDE
from models.autoencoder import DenseAutoencoder
from models.flows import CondMAF, MAF


class AbstractAnalysis(ABC):
    """Abstract base class for analysis.
    
    Any analysis class based on this template must implement the following:
        'dataset' property: Dictionary of datasets 
        'pae' property: Instance of 'Pae'
        'preprocessing' method: does the data loading and preprocessing
        'training' method: trains the Pae model
        'evaluate' method: test the Pae and produce summary plots 
    """

    @abstractproperty
    def dataset(self):
        pass

    @abstractproperty
    def pae(self):
        pass

    @abstractmethod
    def reweighting(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    ESTIMATORS = {
        'gmm': GMM,
        'fftkde': ConvKDE
    }

    FLOWS = {
        'condmaf': CondMAF,
        'maf': MAF
    }

    AUTOENCODERS = {
        'dense': DenseAutoencoder
    }