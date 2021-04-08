"""Concrete implementations of DataLoaders"""

import json
from functools import reduce
from collections import Counter, ChainMap
from operator import add

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from pae.utils import load_json
from . import FEATURE_SETS, SCALERS
from .base import BaseDataloader, BaseDatasetBuilder


class LhcoRnDLoader(BaseDataloader):
    """Data Loader for the clustered LHCO R&D dataset.

    Features data loading from multiple files (hdf), rescaling using
    Sci-kit scalers and custom dataset creation with fine control over the
    amount of events from each file are in the final output
    """
    def __init__(self, file_paths: dict,
                 features: str,
                 scaler: callable = None,
                 exclude: dict = {'scalar': 'mjj'},
                 exclude_range: list = [(None, None)]):
        """Creates an instance of 'LhcoRnDLoader'

        Args:
          file_paths: Dictionary of file path string values and their key 
            labels
          scaler: An instance of a scaler from 'scikit.preprocessing' module
          features: A key from the FEATURE_SETS dict
          exclude: List of features to not be rescaled or introduced in the
              training data. These features of the training data will be outputed
              separately by the 'make_train_val' and 'make_test' methods   
        """
        self._file_paths = file_paths
        self._scaler = scaler
        self.exclude_range = exclude_range
        self.load_datasets(features, exclude=exclude)
        self.train_indexes = {}

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    def load_datasets(self, features, exclude):
        """Loads the requested features from the datasets

        This is usually called by the constructor but cand later on be 
        called again if different features are requested

        Args:
            features: A key from the FEATURE_SETS dictionary
            exclude: List of features to not be rescaled or introduced int the
              training data. These features of the training data will be outputed
              separately by the 'make_train_val' and 'make_test' methods 
        """
        # Select the set of features
        columns = FEATURE_SETS[features]
        self._exclude_keys = exclude

        # Read files and create dataframe dictionary
        self._dataframes = {
            key:pd.read_hdf(data_path)[columns]
            for (key, data_path)
            in self._file_paths.items()
        }

        # Cut based on excluded variables
        for (key, df) in self._dataframes.items():
            for col, rang in zip(exclude, self.exclude_range):
                cmin, cmax = rang[0], rang[1]
                low_out = df[col] < cmin if cmin else np.repeat(False, len(df.index))
                high_out = df[col] > cmax if cmin else np.repeat(False, len(df.index))
                out = df[low_out | high_out].index
                self._dataframes[key] = df.drop(out, axis=0)
        # Drop excluded column and stored them separately
        self._excluded = {}
        for (key, df) in self._dataframes.items():
            self._excluded[key]=df[exclude].to_numpy()
            self._dataframes[key]= df.drop(exclude, axis=1)
    
    def preprocessing(self, fit_key: str):
        """Applies scaling if scaler is defined, otherwise converts to numpy
        
        Args:
          fit_key: Key of the dataset to be used for the scaler training
        """

        # Convert to numpy in absence of scaler
        if self.scaler is None:
            self._scaled_data = {
                key:df.to_numpy() for (key, df) in self._dataframes.items()
            }
        else:
            # Train the scaler on data from 'fit_key'
            scaled0 = self.scaler.fit_transform(self._dataframes[fit_key])

            # Apply transnform to the other datasets
            self._scaled_data = {
                    key:self.scaler.transform(df) 
                    if key is not fit_key 
                    else scaled0
                for (key, df)
                in self._dataframes.items()
            }
    
    def make_train_val(self, data_ratios: dict, sample_size=None, val_split=0,
                       shuffle= False):
        """ Makes dataset of 'sample_size' using 'data_ratios'.

        Args:
            sample_size: The total nuber of events between traning and 
              validation
            val_split: Real value in (0,1) representing the fraction of 
              validation data
            data_ratios: Dictionary of dataset label keys and fraction values. 
              The values should add up to 1 if they are floats otherwise they
              will be interpreded as event counts. The keys should point to 
              loaded datasets.
        """
        # Check if 'data_ratios' is valid
        counts = self._check_data_ratios(sample_size, data_ratios)

        # Create dictionary subsample data indexes
        self.train_indexes = {
            key: np.random.choice(np.arange(self._scaled_data[key].shape[0]),
                                  size=count, replace=False)
                 if shuffle else np.arange(self._scaled_data[key].shape[0]) \
                                 [:count] 
            for (key, count) 
            in counts.items()
        }

        # Add the subsamples to create the traing set
        full_set = np.concatenate([
            self._scaled_data[key][idx] 
            for (key, idx)
            in self.train_indexes.items()
        ])

        # Add together excluded columns of training set
        excluded = np.concatenate([
            self._excluded[key][idx] 
            for (key, idx)
            in self.train_indexes.items()
        ])

        # Split into training and validation, if needed
        if val_split > 0 :
            x_train, x_test, y_train, y_test = train_test_split(
                full_set, excluded, test_size=val_split)
        
        else:
            x_train, x_test, y_train, y_test = full_set, None, excluded, None

        # Create and return output dict
        output = {
            'x_train': x_train,
            'x_valid': x_test,
        }
        # Add the excluded features
        for i,key in enumerate(self._exclude_keys):
            output[key+'_train'] = y_train[:,i]
            output[key+'_valid'] = y_test[:,i] if y_test is not None else None
        return output

    def make_test(self, data_ratios, sample_size=None, replace=True, shuffle=False):
        """ Makes dataset of 'sample_size' using 'data_ratios'

        Args:
            sample_size: The total nuber of test events
            data_ratios: Dictionary of dataset label keys and fraction values. 
              The values should add up to 1 and the keys should point to 
              loaded datasets
            replace: If 'True' the test set could contain events already
             selected for training or validation
        """

        # Check validity of data ratios
        counts = self._check_data_ratios(sample_size, 
                                         data_ratios, 
                                         test_replace=replace)

        # Function to filter out events from training set
        def f(key):
            x = np.arange(self._scaled_data[key].shape[0])
            if not replace and key in self.train_indexes.keys():
                return np.setdiff1d(x,self.train_indexes[key])   
            else: 
                return x

        # Create dictionary subsample data indexes
        self.test_indexes = {
            key: np.random.choice(f(key), size=count, replace=False)
                 if shuffle else f(key)[:count] 
            for (key, count) 
            in counts.items()
        }

        # Add the subsamples to create the traing set
        full_set = np.concatenate([
            self._scaled_data[key][idx] 
            for (key, idx)
            in self.test_indexes.items()
        ])

        # Add together excluded columns of training set
        excluded = np.concatenate([
            self._excluded[key][idx] 
            for (key, idx)
            in self.test_indexes.items()
        ])

        # Save labels
        labels = np.concatenate(
            [np.repeat(key, value) for (key,value) in counts.items()]
        )

        # Construct and return output
        output = {
            'x_test': full_set,
            'labels': labels
        }
        for i,key in enumerate(self._exclude_keys):
            output[key+'_test'] = excluded[:,i]

        return output

    def make_full_dataset(self, train_ratios, test_ratios, fit_key=None, 
                     train_size=None, test_size=None, val_split=0, 
                     shuffle=False, replace=False, random_state=None):
        """Runs the entire data preprocessing chain.
        
        First rescales all data training the scaler on the set given by 
        'fit key', then creates the train and test datasets using the methods 
        'make_train_val' and 'make_test', and lastly merges everything into a
        single dictionary that gets returned.
        """
        # Train scaler and fit all data loaded
        self.preprocessing(fit_key)

        # Create training dataset
        train = self.make_train_val(train_ratios, sample_size=train_size,
                                    val_split=val_split, shuffle=shuffle)

        # Create test dataset
        test = self.make_test(test_ratios, sample_size=test_size,
                                shuffle=shuffle, replace=replace)

        # Merge train and test dictonaries and return the output
        dataset = {**train, **test}
        del train, test
        return dataset

    def _check_data_ratios(self, sample_size, 
                           data_ratios: dict, 
                           test_replace = False):
        """Check the validity of data ratios given sample size"""

        # Convert data ratios to nuber of events
        events_counts = {}
        for (key, fraction) in data_ratios.items():
            # Avoid events already included in the train set
            if test_replace and key in self.train_indexes.keys():
                used_counts = self.train_indexes[key].shape[0] 
            else:
                used_counts = 0
            if fraction == 'all':
                count = self._scaled_data[key].shape[0] - used_counts
            elif fraction <=1:
                if sample_size:
                    count = int(fraction*sample_size)
                else:
                    ValueError(f"Total sample size should be given when " +
                                "using relative data ratios")
            elif fraction >1:
                count = fraction
            # Check if theere is enough data avalilable
            if count + used_counts <= self._scaled_data[key].shape[0]:
                events_counts[key] = count
            else:
                raise ValueError(f"Not enough events in dataset '{key}'")
        return events_counts

class ScalarLoaderLHCO(BaseDataloader):
    """Data loader for LHCO scalar features.
    
    Attributes:
        name (str): Name attributed to the data.
        scaler (callable): Data rescaling transfomation object
            (usually from scikit.preprocessing module) 
    """

    def __init__(self, file_paths, features="all", scaler=None, name=None):
        """Creates a dataloader for LHCO scalar data.
        
        Args:
            file_paths (dict): Dictonary of data label keys and 
                file path values.
            features (str or list(str)): Features to be selected
                from the data files
            scaler (callable): Data rescaling transfomation object
                (usually from scikit.preprocessing module)
            name (str): string literal name of instance (used by the
                DatasetBuilder for labeling)
        
        Raises:
            ValueError: if 'features' argument is not a list nor a string
        """
        super().__init__(file_paths, scaler, name)

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
                                   stop=size)[self.__features]
                self._events[key] = data
                self._available_events[key] = np.ones(len(data)).astype(bool)
        elif sample_sizes is None:
            for key, path in self._file_paths.items():
                data = pd.read_hdf(path)[self.__features]
                self._events[key] = data
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
        else:
            dataset_keys.remove(fit_key)
            self._events[fit_key] = self.scaler.fit_transform(
                                                self._events[fit_key])
            for key in dataset_keys:
                self._events[key] = self.scaler.fit_transform(
                                                self._events[key])

    def get_data(self, indices, exhaust=True):
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
            events_selected = [self._events[label][idx] 
                               for label, idx 
                               in data_indexes.items()]
            if exhaust:
                for label, idx in data_indexes.items():
                    self._available_events[label][idx] = False 
            if self.name is not None:
                dataset[f'{self.name}_{key}'] = np.concatenate(events_selected)
            else:
                dataset[key] = np.concatenate(events_selected)
        return dataset


    
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
        self._data_loaded = False

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
        self._data_loaded = True

    def make_dataset(self, train=None, test=None, validation_split=0.2, 
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
            train = {key: int((1-validation_split)*val) 
                     for key, val 
                     in train.items()}
        else:
            valid = None

        dataset_spec = {k:v for k,v 
                        in zip(['train', 'test', 'valid'], 
                                [train, test, valid])
                        if v}

        all_counts = self.__counts_for_datasets(dataset_spec)
        all_indices = self.__get_event_indexes(all_counts, replace, shuffle)
        

        dataset_indices = {}
        for key, dataset in dataset_spec.items():
            indices = {}
            for k, v in dataset.items():
                indices[k], all_indices[k] = \
                    all_indices[k][:v], all_indices[k][v:]
            dataset_indices[key] = indices

        dataset = [loader.get_data(dataset_indices, exhaust=exhaust) 
                   for loader in self.loaders]
        self.__check_loader_counts()

        return dict(ChainMap(*dataset))

    def __check_names(self):
        for i, loader in enumerate(self.loaders):
            if loader.name is None:
                loader.name = f"DataLoader_{i:02d}"

    def __check_loader_counts(self):
        available = [loader._available_events for loader in self.loaders]
        if all(ele == available[0] for ele in available):
            self._available_events = available[0]
        else:    
            raise RuntimeError("Different number of available events across "
                               "dataloaders")
        
    def __get_event_indexes(self, dataset_spec, replace, shuffle):
        self.__check_loader_counts()
        indices = {}
        for key, count in dataset_spec.items():
            if sum(self._available_events[key]) >= count:
                idx_all = np.where(self._available_events[key] == True)[0]
                if shuffle:
                    idx = np.random.choice(idx_all, count, replace=replace)
                else:
                    idx = idx_all[:count]
                indices[key] = idx
            else:
                raise ValueError("Not enough events to create dataset, try using "
                                 "'replace = True'")
        return indices

    def __counts_for_datasets(self, dataset_spec):
        return dict(reduce(add,map(Counter, dataset_spec.values()))) 