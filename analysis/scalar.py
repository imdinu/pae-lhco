"""Pae analysis workflow for working with scalar features."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon

from analysis.base import AbstractAnalysis
from density.estimators import GMM, ConvKDE
from models.autoencoder import DenseAutoencoder
from models.flows import CondMAF, MAF
from loaders.LHCO import LhcoRnDLoader
from loaders import SCALERS
from plotting import latent_space_plot, loss_plot

class HLFAnalysis(AbstractAnalysis):
    """Pae analysis workflow on LHCO high level features"""

    def __init__(self, pae, dataset=None, loader=None):
        """Creates a HLFAnalysis object.
        
        Args:
            dataset: Dictionary of datasets returned by a LhcoRnDLoader object
            pae: instance of Pae class
        """
        
        self.devices = tf.config.list_physical_devices()
        self.pae = pae
        if dataset:
            self.dataset = dataset
        if loader:
            self.loader = loader
        self.c_inputs = None

    
    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        if isinstance(loader, LhcoRnDLoader):
            self._loader = loader
        else:
            raise ValueError("loader must be an instance of LhcoRnDLoader")

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if isinstance(estimator, str):
            self._estimator = self.ESTIMATORS[estimator]()
        else:
            self._estimator = estimator

    @property
    def pae(self):
        return self._pae

    @pae.setter
    def pae(self, pae):
        self._pae = pae

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if isinstance(dataset, dict):
            self._dataset = dataset
        else:
            raise ValueError("dataset must be a python dictionary")

    def make_dataset(self, **kwargs):
        """Creates the dataset by passing the kwargs to the loader"""
        
        self.dataset = self.loader.make_full_dataset(**kwargs)

    def reweighting(self, estimator=None, fit_key=None):
        """Computes event weights based on mjj.

        Args:
            estimator: Instance of density estimator or string.
            fit_key: Dict key pointing to the dataset to be used for estimator 
                     training.
        """
        if estimator:
            self.estimator = estimator
        if fit_key:
            self.estimator.fit(self.dataset[fit_key])

        mjj_keys = ['mjj_train', 'mjj_test', 'mjj_valid']
        eval_keys = [key for key in mjj_keys if key in self.dataset.keys()]
        self.weights= {key.split("_")[1]: 
                       self.estimator.get_weights(self.dataset[key])
                       for key in eval_keys}

    def make_cond_inputs(self, c_keys, scaler='min_max', **kwargs):
        """Generates conditinal input for normalizing flows. 
        
        Args:
            fit_key: Tuple of dict keys pointing to the datasets to be used as 
                     conditional inputs for the normalizing flows. Will be
                     interpreted as (key_c_train, key_c_test ,key_c_valid)
        """
        self.c_scaler = SCALERS[scaler](**kwargs)
        key_c_train, key_c_test, key_c_valid = c_keys

        self.c_inputs = {}

        c_train = self.c_scaler.fit_transform(self.dataset[key_c_train] \
                                                .reshape(-1,1))
        c_test = self.c_scaler.transform(self.dataset[key_c_test] \
                                                .reshape(-1,1))
        self.c_inputs['train'] = c_train
        self.c_inputs['test'] = c_test
        if key_c_valid:
            c_valid = self.c_scaler.transform(self.dataset[key_c_valid] \
                                              .reshape(-1,1))
            self.c_inputs['valid'] = c_valid


    def train(self, ae_train, nf_train, device=None):
        """Trains the pae model using configuration dictionaries.
        
        Args:
            ae_train: Dictionary of kwargs passed to the 'fit' funtion of the 
                      autoencoder
            nf_train: Dictionary of kwargs passed to the 'fit' funtion of the 
                      normalizing flow
        """
        print(self.devices)
        if self.weights:
            ae_train['sample_weight'] = self.weights['train']
            if 'x_valid' in self.dataset.keys():
                ae_valid = (self.dataset['x_valid'], 
                            self.dataset['x_valid'],
                            self.weights['valid'])
                ae_train['validation_data'] = ae_valid

        if self.c_inputs:
            c = self.c_inputs['train']
            if 'x_valid' in self.dataset.keys():
                nf_train['validation_data'] = [self.dataset["x_valid"], 
                                            self.c_inputs['valid']]
        else:
            c = None
            if 'x_valid' in self.dataset.keys():
                nf_train['validation_data'] = [self.dataset["x_valid"], None]
        
        self.pae.fit(self.dataset["x_train"], c, 
                                    ae_train, nf_train)
        
        if 'x_valid' in self.dataset.keys():
            self.pae.compute_implicit_sigma(self.dataset['x_valid'])
        else :
            self.pae.compute_implicit_sigma(self.dataset['x_train'])

    def evaluate(self, prc, sig_label='sig', js_bins=60):
        """Evaluate the pae in terms of sig efficiency and mass sculpting"""

        # Anomaly score on the training set
        c = self.c_inputs['train'] if self.c_inputs else None
        ascore_train = -self.pae.anomaly_score(self.dataset['x_train'], c=c)

        # Apply percentile cut
        x_prc = np.percentile(ascore_train, prc)
        i_prc = (ascore_train >= x_prc)

        # Compute JS divergence on training set
        mjj = self.dataset['mjj_train']
        n_full, b = np.histogram(mjj, bins=js_bins, density=True)
        n_prc, _ = np.histogram(mjj[i_prc], bins=b, density=True)

        js_div_train = jensenshannon(n_full,n_prc)

        # Anomaly score on the test set
        c = self.c_inputs['test'] if self.c_inputs else None
        ascore_test = -self.pae.anomaly_score(self.dataset['x_test'], c=c)

        # Apply percentile cut
        x_prc = np.percentile(ascore_test, prc)
        i_prc = (ascore_test >= x_prc)

        # Compute JS divergence on training set
        mjj = self.dataset['mjj_test']
        n_full, b = np.histogram(mjj, bins=js_bins, density=True)
        n_prc, _ = np.histogram(mjj[i_prc], bins=b, density=True)

        js_div_test = jensenshannon(n_full,n_prc)

        # Binarize labels based on 'sig_label'
        def binarize(label):
            return 1 if label == sig_label else 0
        labels = np.array(list(map(binarize, self.dataset['labels'])))
        sig_label = (labels==1)
        bkg_label = (labels==0)

        # Compute signal efficiecy and background rejection
        sig_eff = sum(i_prc&sig_label)/sum(sig_label)
        bkg_rej = 1-sum(i_prc&bkg_label)/sum(bkg_label)

        # Compute AUC on the test set
        fpr, tpr, _ = roc_curve(labels, ascore_test)
        test_auc = auc(1-fpr, tpr)

        self.results = {
            'js_div_train': js_div_train,
            'js_div_test': js_div_test,
            'sig_eff': sig_eff,
            'bkg_rej': bkg_rej,
            'auc': test_auc
        }

        return self.results

    def plot_training(self, filename=None):

        fig = loss_plot(self.pae.history)
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')

    def plot_latent_space(self, filename=None):

        z_true = self.pae.ae.encode(self.dataset['x_train'])
        c = self.c_inputs['train'] if self.c_inputs else None
        z_sample = self.pae.nf.sample(c, self.dataset['x_train'].shape[0])

        fig = latent_space_plot(z_true, z_sample)
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')


        

