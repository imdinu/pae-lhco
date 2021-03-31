"""Defines the base abstractions for building DataLoaders."""

from abc import ABC, abstractmethod, abstractproperty

from utils import load_json
from loaders import FEATURE_SETS, SCALERS

class AbstractDataloader(ABC):
    """Abstract base class of a dataloader.

    Any data Loader Must implement the 'scaler' attribute, which is usually
    choosen from 'sklearn.preprocessing'. 

    The three methods of this class that require to be implemented  
    are: 'load_datasets', 'make_train_val' and 'make_test'
    """

    @abstractproperty
    def scaler(self):
        pass

    @abstractmethod
    def load_datasets(self):
        pass

    @abstractmethod
    def make_train_val(self):
        pass

    @abstractmethod
    def make_test(self):
        pass

    @classmethod
    def from_json(cls, json_file):
        """Creates an instance of 'LhcoRnDLoader' based on a json file"""
        kwargs = load_json(json_file)
        scaler = kwargs['scaler']
        if scaler is not None:
            kwargs['scaler'] = SCALERS[scaler](**kwargs['scaler_kwargs'])
        del kwargs["scaler_kwargs"]
        return cls(**kwargs)
    