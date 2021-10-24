from pathlib import Path
import os

import numpy as np
import tensorflow as tf

from pae.utils import load_json
from pae.models import OPTIMIZERS, ACTIVATIONS, REGULARIZERS, MODELS, CALLBACKS
from pae.models.flows import nf_loss_bootstrap

class Pae():
    """Probabilistic Autoencoder network architecture
    
    This is a complex model involving an Autoencoder and Normalizing Flow
    You can use the constructor to create an instance, provided that you
    already built and *compiled* an Autoencoder and a Flow model. 

    Another option is to use an instance of PaeBuilder to create the
    entire architecture step by step.
    """
    def __init__(self, ae=None, nf=None):
        """Creates a PAE given an Autoencoder and a Flow model"""
        self.history = {}
        self.is_fit = False
        self.ae = ae
        self.nf = nf
        self.sigma_square = None

    @property
    def nf(self):
        return self._nf

    @nf.setter
    def nf(self, nf):
        self._nf = nf

    @property
    def ae(self):
        return self._ae
    
    @ae.setter
    def ae(self, ae):
        self._ae = ae

    def compute_implicit_sigma(self, x_val):
        """Calculates the sigma parameter on the given data and stores it"""

        self.sigma_square = self.reco_error(x_val, axis = 0)

    def reco_error(self, x_true, axis=1):
        """Returs the autoencoder reconstruction error of the input data"""

        x_reco = self._ae(x_true)
        return np.mean(np.square(x_true-x_reco), axis=axis)

    def log_prob_encoding(self, x, c=None):
        """Return the log likelihood of a given encoding"""
        
        z = self._ae.encode(x)
        if c is not None:
            return self._nf([z, c]).numpy()
        else:
            return self._nf(z).numpy()

    def anomaly_score(self, x, c=None):
        """Calculates the anomaly scores for the input data"""

        reco_error = np.square(self._ae(x)-x)
        z = self._ae.encode(x)
        if c is not None:
            byz = self._nf.inverse([z, c])
            detJ = self._nf.inverse_log_det_jacobian([z, c])
        else:
            byz = self._nf.inverse(z)
            detJ = self._nf.inverse_log_det_jacobian(z)

        return -0.5*np.dot(reco_error,self.sigma_square**(-1)) - \
                0.5*np.linalg.norm(byz,axis=1)**2 + detJ

    def fit_ae(self, ds_train, **kwargs):
        self.history['ae'] = self._ae.fit(ds_train, **kwargs)

    def fit_nf(self, ds_train, **kwargs):
        self.history['nf'] = self._nf.fit(ds_train, **kwargs)

    def fit(self, x, c=None, kwargs_ae=None, kwargs_nf=None):
        """Trains the autoencoder and then the flow model"""
        
        # Autoencoder training
        self.history['ae'] = self._ae.fit(x=x, 
                                      y=x,  
                                      **kwargs_ae)

        # Encode training and validation data
        z = self.ae.encode(x)

        # Check if there are conditional inputs
        if c is not None:
            if 'validation_data' in kwargs_nf.keys():
                x_valid, c_valid = kwargs_nf['validation_data']
                kwargs_nf['validation_data'] = ([self.ae.encode(x_valid), c_valid], np.zeros(c_valid.shape[0]))
            
            #Train Normalizing Flow
            self.history['nf']= self._nf.fit(x=[z, c],
                                            y=np.zeros(z.shape),
                                            **kwargs_nf)
        else:
            if 'validation_data' in kwargs_nf.keys():
                x_valid, _ = kwargs_nf['validation_data']
                kwargs_nf['validation_data'] = (self.ae.encode(x_valid), np.zeros(x_valid.shape[0]))
            
            #Train Normalizing Flow
            self.history['nf']= self._nf.fit(x=z,
                                            y=np.zeros(z.shape),
                                            **kwargs_nf)
        self.is_fit = True

    def save_model(self, dir=None):
        """Saves the AE and NF models in the given directory.

        Args:
            dir: path of the directory where the models will be saved

        Returns:
            None
        """

        raise NotImplementedError("Functionality not yet available")
        # ae_path = Path(dir) / "ae/"
        # nf_path = Path(dir) / "nf/"
        # if not os.path.isdir(ae_path):
        #     os.mkdir(ae_path)
        # if not os.path.isdir(nf_path):
        #     os.mkdir(nf_path)
        # self.ae.save(ae_path)
        # self.nf.save(nf_path)

    def load_model(self, dir=None):
        """Loads the AE and NF models from the given directory.
        
        Args:
            dir: path of the directory containing the models

        Returns:
            None
        """
        raise NotImplementedError("Functionality not yet available")
        # ae_path = Path(dir) / "ae/"
        # nf_path = Path(dir) / "nf/"
        # self.ae = tf.keras.models.load_model(ae_path)
        # custom_objects = {"nf_loss_bootstrap": nf_loss_bootstrap}
        # with tf.keras.utils.custom_object_scope(custom_objects):
        #     self.nf = tf.keras.models.load_model(nf_path)


class PaeBuilder():
    """Builder class for PAE objects.
    
    Creates a PAE step by step given the configuration for the various parts.
    You need to provide a configurations for evey one of the four the methods:
    'make_ae_model', 'make_ae_optimizer', 'make_nf_model' and
    'make_nf_optimizer'. After that you can just call the getter for the 'pae'
    property.

    *Warning:* After calling the 'pae' getter, the entire builder will reset. 
    Make sure you asign the getter output to a variable.
    """
    def __init__(self):
        """Creates an unconfigured PaeBuilder."""
        self.reset()

    def reset(self):
        """Resets the builder to initial state"""
        self._ae = None
        self._nf = None
        self._optim_ae = 'Adam'
        self._optim_nf = 'Adam'
        self._compiled_ae = False
        self._compiled_nf = False

    @classmethod
    def from_json(cls, json_file):
        """Creates a Pae object from a configuration file"""
        pae_config = json_file if isinstance(json_file, dict) \
                            else load_json(json_file)
        ae_config = {key.split(':')[1]:pae_config.pop(key) 
            for key in list(pae_config.keys()) 
            if 'AE:' in key}
        nf_config = {key.split(':')[1]:pae_config.pop(key) 
                    for key in list(pae_config.keys()) 
                    if 'NF:' in key}

        builder = cls()

        builder._interpret_args(pae_config)
        builder._interpret_config(ae_config)
        builder._interpret_config(nf_config)

        ae_model = pae_config.pop('ae_model')
        ae_optim = pae_config.pop('ae_optimizer')

        nf_model = pae_config.pop('nf_model')
        nf_optim = pae_config.pop('nf_optimizer')

        ae_train = {key.split('_', 1)[1]:pae_config.pop(key) 
                    for key in list(pae_config.keys()) 
                    if 'ae' in key}
        nf_train = {key.split('_', 1)[1]:pae_config.pop(key) 
                    for key in list(pae_config.keys()) 
                    if 'nf' in key}
        if pae_config:
            raise Warning("Json config contains unusable kwargs", 
                          pae_config.keys())

        builder.make_ae_model(ae_model, ae_config)
        builder.make_nf_model(nf_model, nf_config)
        builder.optimizer_ae = ae_optim
        builder.optimizer_nf = nf_optim
        builder.compile_ae()
        builder.compile_nf()

        return builder.pae, ae_train, nf_train


    @property
    def pae(self):
        """Returns the built Pae object and restes builder"""
        if not self._compiled_ae:
            raise RuntimeError('Autoencoder is not compiled. Use '
                               'builder.compile_ae(cls, kwargs_dict)')
        elif not self._compiled_nf:
            raise RuntimeError('Normalizing flow model is not compiled. Use '
                               'builder.compile_nf(cls, kwargs_dict)')
        pae = Pae(self._ae, self._nf)
        self.reset()
        return pae

    @property
    def optimizer_ae(self):
        return self._optim_ae
    
    @optimizer_ae.setter
    def optimizer_ae(self, optimizer_ae):
        self._optim_ae = optimizer_ae

    @property
    def optimizer_nf(self):
        return self._optim_nf
    
    @optimizer_nf.setter
    def optimizer_nf(self, optimizer_nf):
        self._optim_nf = optimizer_nf

    def make_ae_model(self, cls, kwargs_dict):
        """Creates Autoencoder based on 'kwargs_dict'"""
        ae = cls(**kwargs_dict)
        self._ae = ae

    def make_nf_model(self, cls, kwargs_dict):
        """Creates Flow model based on 'kwargs_dict'"""
        nf = cls(**kwargs_dict)
        self._nf = nf

    def make_ae_optimizer(self, cls, kwargs_dict):
        """Creates optimizer for Autoencoder based on 'kwargs_dict'"""
        optimizer = cls(**kwargs_dict)
        self._optim_ae = optimizer
    
    def make_nf_optimizer(self, cls, kwargs_dict):
        """Creates optimizer for Flow model based on 'kwargs_dict'"""
        optimizer = cls(**kwargs_dict)
        self._optim_nf = optimizer

    def compile_ae(self, loss='mse'):
        """Compile the Autoencoder model using the given loss function"""
        if self._ae is None:
            raise RuntimeError('Autoencoder is not set. Use '
                               'builder.make_ae_model(cls, kwargs_dict)')
        self._ae.compile(self._optim_ae, loss)
        self._compiled_ae = True

    def compile_nf(self, loss=nf_loss_bootstrap):
        """Compile the Flow model using the given loss function"""
        if self._optim_nf is None:
            raise RuntimeError('Normalizing flow model not set. Use '
                               'builder.make_nf_model(cls, kwargs_dict)')
        self._nf.compile(self._optim_nf, loss)
        self._compiled_nf = True

    def _interpret_args(self, config):
        """Translates strings to objects passing kwargs constructiors"""
        for key in list(config):
            if 'model' in key:
                config[key] = MODELS[config[key]]
            elif 'optimizer' in key:
                if key+'_kwargs' in config:
                    config[key] = OPTIMIZERS[config[key]](**config.pop(
                                                            key+"_kwargs")
                                                        )
                elif 'kwargs' not in key:
                    config[key] = OPTIMIZERS[config[key]]()
            elif 'callbacks' in key and 'kwargs' not in key:
                if key+'_kwargs' in config:
                    config[key] = [CALLBACKS[callback](**kwargs)
                                    for callback, kwargs  
                                    in zip(config[key],config.pop(
                                                            key+"_kwargs")
                                                        )
                                    ]
                else:
                    config[key] = [CALLBACKS[callback]()
                                    for callback in config[key]
                                    ] 
    def _interpret_config(self, config):
        for key in config:
            if 'activation' in key:
                config[key]= ACTIVATIONS[config[key]]
            if 'reg' in key:
                if len(config[key]) == 2 :
                    config[key]= REGULARIZERS['l1_l2'](**config[key])
                elif len(config[key]) == 1:
                    config[key]= REGULARIZERS[list(config[key])[0]](**config[key])
                else:
                    config[key]=None