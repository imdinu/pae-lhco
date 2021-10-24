"""Module of base classes for data management tools.

Defines the base abstractions for building DataLoaders, and dataset builders
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import TransformerMixin

from pae.utils import load_json
from . import FEATURE_SETS, SCALERS


class BaseDataloader(ABC):
    """Abstract base class for a dataset loader

    Classes inheriting from BaseDataloader must implmement the following
    methods:
        - 'load_files'
        - 'rescale'
        - 'make_dataset'
    """
    @abstractmethod
    def __init__(self, file_paths, scaler, name):
        """ Initialize data loader.
        
        Args:
            file_paths (dict): Dictonary of data label keys and 
                file path values.
            scaler (callable): Data rescaling transfomation object
                (usually from scikit.preprocessing module)
            name (str): string literal name of instance (used by the
                DatasetBuilder for labeling)
        
        """
        self._file_paths = file_paths
        self.name = name
        self._events = {}
        self._available_events = {}
        self.scaler = scaler

    def __len__(self):
        return sum(len(x) for x in self._events.values())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._events[key]
        elif isinstance(key, dict):
            return np.concatenate([self._events[label][idx] 
                                   for label, idx 
                                   in key.items()])
        else:
            raise KeyError(key)

    @abstractmethod
    def load_events(self):
        pass

    @abstractmethod
    def rescale(self):
        pass

    def make_dataset(self, indices, exhaust=False, kfold=None):
        """Creates a dataset using the previously loaded data based on indices.

        Args:
            indices (dict): A nested dictionary of string literal keys and 
                values which are, in turn, dictionaries of index values.
            exhaust (bool): weather or not to remove the event indices from 
                the available events pool
            kfold (int): Number of 
                
        Returns:
            Dictionary of samples labeled acording to indices keys and 
            instance name
        """

        dataset = {}
        for key, data_indexes in indices.items():
            events_selected = self[data_indexes]
            if exhaust:
                for label, idx in data_indexes.items():
                    self._available_events[label][idx] = False 
            if self.name is not None:
                dataset[f'{self.name}_{key}'] = events_selected
            else:
                dataset[key] = events_selected
        return dataset

    @property
    def scaler(self):
        """Data rescaler instance from scikit.preprocessing module."""
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        if isinstance(scaler, str):
            try:
                self._scaler = SCALERS[scaler]()
            except KeyError:
                print(f"'{scaler}' is not a known scaler description. "
                      f"Available options are {list(SCALERS.keys())}")
                raise
        elif isinstance(scaler, (TransformerMixin, type(None))):
            self._scaler = scaler
        else:
            raise TypeError("Scaler must either be a scikit-learn.preprocessing"
                        "transflormation or a string literal description of one")


    @classmethod
    def from_json(cls, path):
        """Instantiates a dataloader based on kwargs from a json file.
        
        Args:
            path (Path): Path to the json_file

        Returns:
            Instance of data loader
        """
        kwargs = load_json(path)

        if 'scaler' in kwargs:
            scaler = kwargs.pop('scaler')
            if 'scaler_kwargs' in kwargs.keys():
                sclaer_kwargs = kwargs.pop('scaler_kwargs')
                if scaler is not None:
                    scaler = SCALERS[scaler](**sclaer_kwargs)
            elif scaler is not None:
                scaler = SCALERS[scaler]
            obj = cls(**kwargs)
            obj.scaler = scaler
        else:
            obj = cls(**kwargs)

        return obj

class BaseDatasetBuilder(ABC):
    """Abstract base class of a dataset builder.

    Dataset builders inheriting from this base class must implement the
    following methods:
        - data_preparation: loads and rescales the data
        - make_datasets: creates an output dataset
    """

    @abstractmethod
    def data_preparation(self):
        pass

    @abstractmethod
    def make_dataset(self):
        pass

              
    # @classmethod
    # def from_json(cls, path):
    #     kwargs = load_json(path)
        
    #     scaler = kwargs.pop('scaler')
    #     if 'scaler_kwargs' in kwargs.keys():
    #         sclaer_kwargs = kwargs.pop('scaler_kwargs')
    #         scaler = SCALERS[scaler](**sclaer_kwargs)
    #     else:
    #         scaler = SCALERS[scaler]
    #     obj = cls(**kwargs)
    #     obj.scaler = scaler
    #     return obj
    