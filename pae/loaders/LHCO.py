"""Concrete implementations of DataLoaders"""

import json
from functools import reduce
from collections import Counter, ChainMap
from operator import add

import h5py
import pandas as pd
import numpy as np

from pae.utils import load_json
from . import FEATURE_SETS, SCALERS
from .base import BaseDataloader, BaseDatasetBuilder


class ScalarLoaderLHCO(BaseDataloader):
    """Data loader for LHCO scalar features.
    
    Attributes:
        name (str): Name attributed to the data.
        scaler (callable): Data rescaling transfomation object
            (usually from scikit.preprocessing module) 
    """

    def __init__(self, file_paths, features="all", scaler=None, 
                 nan=0., inf=0, name=None):
        """Creates a dataloader for LHCO scalar data.
        
        Args:
            file_paths (dict): Dictonary of data label keys and 
                file path values.
            features (str or list(str)): Features to be selected
                from the data files
            scaler (callable): Data rescaling transfomation object
                (usually from scikit.preprocessing module)
            inf (int or float): value to replace 'inf' in data
            nan (int or float): value to replace 'nan' in data
            name (str): string literal name of instance (used by the
                DatasetBuilder for labeling)
        
        Raises:
            ValueError: if 'features' argument is not a list nor a string
        """
        super().__init__(file_paths, scaler, name)

        self.nan = nan
        self.inf = inf

        if isinstance(features, list):
            self.__features = features
        elif isinstance(features, str):
            self.__features = FEATURE_SETS[features]
        else:
            raise ValueError("Feature argument must be a list of column names "
                             "or a string literal identifier of a feature set")

    def load_events(self, sample_sizes=None):
        """Loads events baswed on given sample_sizes.

        This function relies on the `file_paths` attribute for the events'
        location and the `features` attribute for column selection.

        Args:
            sample_size (dict, optional): the number of events to be read.
                if not included the entire file will be loaded

        Returns:
            self: instance of ScalarLoaderLHCO 

        Raises:
            ValueError: if sample size is not of 'NoneType', nor 'dict'
        """

        if isinstance(sample_sizes, dict):
            for key, size in sample_sizes.items():
                data = pd.read_hdf(self._file_paths[key], 
                                   stop=size)[self.__features] \
                                   .fillna(self.nan) \
                                   .replace(np.inf, self.inf)
                self._events[key] = data.to_numpy()
                self._available_events[key] = np.ones(len(data)).astype(bool)
        elif sample_sizes is None:
            for key, path in self._file_paths.items():
                data = pd.read_hdf(path)[self.__features].fillna(self.nan) \
                            .replace(np.inf, self.inf)
                self._events[key] = data.to_numpy()
                self._available_events[key] = np.ones(len(data)).astype(bool)
        else:
            raise ValueError("'sample_sizes' must be a dictionary")
        return self

    def rescale(self, fit_key):
        """Applies rescaling transformation to loaded data.

        The transformation is applied using the 'scalar' property. Note that
        the original events are not kept. In order to restore the initial data
        you can reverse the transformation.

        Args:
            fit_key (str): String literal identifier for the dataset to be
                used for fitting the scaler.
        Returns:
            None
        """
        
        dataset_keys = list(self._events.keys())
        if fit_key not in dataset_keys:
            raise KeyError("Unrecognized value of 'fit_key'")
        dataset_keys.remove(fit_key)
        self._events[fit_key] = self.scaler.fit_transform(
                                            self._events[fit_key])
        for key in dataset_keys:
            self._events[key] = self.scaler.transform(
                                            self._events[key])
        return self

    def make_dataset(self, indices, exhaust=True):
        """Creates a dataset using the previously loaded data based on indices.

        Args:
            indices (dict): A nested dictionary of string literal keys and 
                values which are, in turn, dictionaries of index values.
            exhaust (bool): weather or not to remove the event indices from 
                the available events pool
                
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

class ImageLoaderLHCO(BaseDataloader):
    """Data loader for LHCO jet images.
    
    Attributes:
        name (str): Name attributed to the data.
    """
    def __init__(self, file_paths, name):
        """Creates a dataloader for LHCO jet image data.
        
        Args:
            file_paths (dict): Dictonary of data label keys and 
                file path values.
            name (str): string literal name of instance (used by the
                DatasetBuilder for labeling)
        
        Raises:
            ValueError: if 'features' argument is not a list nor a string
        """
        super().__init__(file_paths, None, name)

    def rescale(self):
        """Applies rescaling transformation to loaded data.

        Returns:
            None
        """
        raise NotImplementedError("Jet images are already scaled")

    def load_events(self, sample_sizes=None):
        """Loads events baswed on given sample_sizes.

        This function relies on the `file_paths` attribute for the events'
        location and the `features` attribute for column selection.

        Args:
            sample_size (dict, optional): the number of events to be read.
                if not included the entire file will be loaded

        Returns:
            self: instance of ScalarLoaderLHCO 

        Raises:
            ValueError: if sample size is not of 'NoneType', nor 'dict'
        """

        if isinstance(sample_sizes, dict):
            for key, size in sample_sizes.items():
                data = h5py.File(self._file_paths[key], 'r')['multijet'][:size]
                self._events[key] = data
                self._available_events[key] = np.ones(len(data)).astype(bool)
        elif sample_sizes is None:
            for key, path in self._file_paths.items():
                data = h5py.File(path, 'r')['multijet']
                self._events[key] = data
                self._available_events[key] = np.ones(len(data)).astype(bool)
        else:
            raise ValueError("'sample_sizes' must be a dictionary")
        return self


    
class DatasetBuilder(BaseDatasetBuilder):
    """ Dataset builder based on one or multiple dataloaders.

    Attributes:
        loaders (list): A list of dataloader objects. 
    """

    def __init__(self, *loaders):
        """Default constructor for a DatasetBuilder.
        
        Creates a dataset builder using a variable number of dataloaders.

        Args:
            *loaders: Variable length DataLoader list
        """
        self.loaders = loaders
        self.__check_names()
        self._available_events = {}
        self._data_loaded = None

    def data_preparation(self, fit_key=None, sample_sizes=None):
        """Loads and rescales the data using all loaders.
        
        Args:
            fit_key (str, optional): string literal pointing to data used for scaler 
                    fitting
            sample_sizes (dict, optional): specification of how much data to 
                    be loaded by each Dataloader instance

        """

        for loader in self.loaders:
            loader.load_events(sample_sizes)
            if fit_key is not None and loader.scaler is not None:
                loader.rescale(fit_key)

        self._available_events = self.loaders[0]._available_events
        self._data_loaded = sample_sizes

    def make_dataset(self, train=None, test=None, validation_split=0, 
                     replace=False, shuffle=True, exhaust=True):
        """Creates a dataset based on configuration dictionary

        Args:
            train (dict): dictionary specification for training dataset
            test (dict): dictionary specification for test dataset
            validation_split (float): the ammount of traning data to be labeled
                as validation
            replace (bool): weather or not to sample with replacement
            shuffle (bool): weather or not to randomply choose indexes rather
                than taking them consecutively
            exhaust (bool): weather or not to make the events selected 
                unavialable to future calls of this function

        Returns:
            Dictionary of datasets.

        Raises:
            ValueError: if argument 'train' is *None* and 'validation_split'
                is not zero 
            RuntimeError: if the available events across all dataloaders are
                not the same
            ValueError: if there are not enough evailable events to create the
                dataset
        """

        if validation_split > 0:
            if not train:
                raise ValueError("Training set must exist if "
                                 "'validation_split' is not zero.")
            valid = {key: int(validation_split*val)
                     for key, val 
                     in train.items()}
            train = {key: val-int(validation_split*val) 
                     for key, val 
                     in train.items()}
        else:
            valid = None

        dataset_spec = {k: v for k, v 
                        in zip(['train', 'test', 'valid'], 
                                [train, test, valid])
                        if v}

        return self._dataset_from_spec(dataset_spec, replace, 
                                        shuffle, exhaust)

    def _dataset_from_spec(self, dataset_spec, replace, shuffle, exhaust):
        all_counts = self.__counts_for_datasets(dataset_spec)
        all_indices = self.__get_event_indexes(all_counts, replace, shuffle)

        dataset_indices = {}
        for key, dataset in dataset_spec.items():
            indices = {}
            for k, v in dataset.items():
                indices[k], all_indices[k] = \
                    all_indices[k][:v], all_indices[k][v:]
            dataset_indices[key] = indices

        dataset = [loader.make_dataset(dataset_indices, exhaust=exhaust) 
                   for loader in self.loaders]
        #self.__check_loader_counts()

        result = dict(ChainMap(*dataset))

        for key, counts in dataset_spec.items():
            result[f'labels_{key}'] = np.concatenate([np.repeat(label, count)
                                                        for label, count 
                                                        in counts.items()])

        return result


    def __check_names(self):
        for i, loader in enumerate(self.loaders):
            if loader.name is None:
                loader.name = f"DataLoader_{i:02d}"

        
    def __get_event_indexes(self, dataset_spec, replace, shuffle):
        # self.__check_loader_counts()
        indices = {}
        for key, count in dataset_spec.items():
            if sum(self._available_events[key]) >= count or replace:
                idx_all = np.where(self._available_events[key] == True)[0]
                if shuffle:
                    idx = np.random.choice(idx_all, count, replace=replace)
                else:
                    idx = idx_all[:count]
                indices[key] = idx
            elif sum(self._available_events[key]) == 0:
                raise RuntimeError(f"All {key} events have already "
                                   "previously been exhausted")
            else:
                raise ValueError("Not enough events to create dataset, "
                                 "try using 'replace = True'")
        return indices

    def __counts_for_datasets(self, dataset_spec):
        return dict(reduce(add,map(Counter, dataset_spec.values()))) 