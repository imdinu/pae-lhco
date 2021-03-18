"""Defines the base abstractions for building DataLoaders."""

from abc import ABC, abstractmethod, abstractproperty

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
    