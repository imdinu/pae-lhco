"""Defines the base abstractions for building DataLoaders."""

from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin

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
    def load_data(self):
        pass

    @abstractmethod
    def make_dataset(self):
        pass

    @property
    def scaler(self):
        return self.scaler

    @scaler.setter
    def scaler(self, scaler):
        if isinstance(scaler, str):
            try:
                self.scaler = SCALERS[scaler]
            except KeyError:
                print(f"'{scaler}' is not a known scaler description. "
                      f"Available options are {list(SCALERS.keys())}")
                raise
        elif isinstance(scaler, TransformerMixin):
            self.scaler = scaler
        else:
            raise TypeError("Scaler must either be a scikit-learn.preprocessing"
                        "transflormation or a string literal description of one")
              
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
    
class BaseDatasetSpec(ABC):
    """Abstract base class for a dataset specification

    """

    @classmethod
    def from_json(cls, Path):
        pass