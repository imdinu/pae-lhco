"""Defines the base abstractions for building DataLoaders."""

from abc import ABC, abstractmethod, abstractproperty

from pae.utils import load_json
from . import FEATURE_SETS, SCALERS

class BaseDataloader(ABC):
    """Abstract base class of a dataloader.

    Any data Loader Must implement the 'scaler' attribute, which is usually
    choosen from the 'sklearn.preprocessing' module, but could also just be 
    'NoneType' 

    The interface of a dataloader is defined by the following methods:  
      - 'load_datasets' 
      - 'make_train_val'
      - 'make_test'
    """


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
    def from_json(cls, path):
        """Instantiates a dataloader based on kwargs from a json file.
        
        Args:
            path (Path): Path to the json_file

        Returns:
            Instance of dataloader
        """
        kwargs = load_json(path)
        scaler = kwargs['scaler']
        if scaler is not None:
            kwargs['scaler'] = SCALERS[scaler](**kwargs['scaler_kwargs'])
        del kwargs["scaler_kwargs"]
        return cls(**kwargs)
    