"""Concrete implementations of DataLoaders"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from loaders import FEATURE_SETS
from loaders.base import AbstractDataloader


class LhcoRnDLoader(AbstractDataloader):
    """Data Loader for the clustered LHCO R&D dataset.

    Features data loading from multiple files (hdf), rescaling using
    Sci-kit scalers and custom dataset creation with fine control over the
    amount of events from each file are in the final output
    """
    def __init__(self, file_paths: dict,
                 features: str,
                 scaler:callable):
        """Creates an instance of 'LhcoRnDLoader'

        Args:
          file_paths: Dictionary of file path string values and their key 
            labels

          scaler: An instance of a scaler from 'scikit.preprocessing' module
          features: A key from the FEATURE_SETS dict
        """
        self._file_paths = file_paths
        self._scaler = scaler
        self.load_datasets(features)
        self.train_indexes = {}

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    def load_datasets(self, features, exclude=['mjj']):
        """Loads the requested features from the datasets

        This is usually called by the constructor but cand later on be 
        called again if different features are requested

        Args:
            features: A key from the FEATURE_SETS dictionary
            exclude: List of features to not be rescaled or introduced int the
              training data. These features of the training data will be outputed
              separately by the 'make_train_val' and 'make_test' methods 
        """
        columns = FEATURE_SETS[features]

        # Read files and create dataframe dictionary
        self._dataframes = {
            key:pd.read_hdf(data_path)[columns]
            for (key, data_path)
            in self._file_paths.items()
        }

        # Drop excluded column and stored them separately
        self._excluded = {}
        for (key, df) in self._dataframes.items():
            self._excluded[key]=df[exclude].to_numpy()
            self._dataframes[key]= df.drop(exclude, axis=1)
    
    def preprocessing(self, fit_key: str):
        """Applies scaling if scaler is defined, otherwise converts to numpy
        
        Args:
          fit_key: Key of the dataset to be used as the scaler training
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
    
    def make_train_val(self, sample_size, data_ratios: dict, val_split=0.):
        """ Makes dataset of 'sample_size' using 'data_ratios'.

        Args:
            sample_size: The total nuber of events between traning and 
              validation
            val_split: Real value in (0,1) representing the fraction of 
              validation data
            data_ratios: Dictionary of dataset label keys and fraction values. 
              The values should add up to 1 and the keys should point to 
              loaded datasets.
        """
        # Check if 'data_ratios' is valid
        counts = self._check_data_ratios(sample_size, data_ratios)

        # Create dictionary subsample data indexes
        self.train_indexes = {
            key: np.random.choice(np.arange(self._scaled_data[key].shape[0]),
                                  size=count, replace=False)
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
            'excl_train': y_train,
            'excl_valid': y_test
        }
        return output

    def make_test(self, sample_size, data_ratios, replace=True):
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
            'excl_test': excluded,
            'labels': labels
        }
        return output

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
            count = int(fraction*sample_size) 
            # Check if theere is enough data avalilable
            if count + used_counts <= self._scaled_data[key].shape[0]:
                events_counts[key] = count
            else:
                raise ValueError(f"Not enough events in dataset '{key}'")
        return events_counts

    
