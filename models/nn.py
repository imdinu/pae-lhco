import numpy as np

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
        self._ae = ae
        self._nf = nf
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

    def log_prob_encoding(self, x):
        """Return the log likelihood of a given encoding"""
        z = self._ae.encode(x)
        return self._nf(z)

    def anomaly_score(self, x):
        """Calculates the anomaly scores for the input data"""
        reco_error = np.square(self._ae(x)-x)
        z = self._ae.encode(x)
        byz = self._nf.inverse(z)
        detJ = self._nf.inverse_log_det_jacobian(z)

        return -0.5*np.dot(reco_error,self.sigma_square**(-1)) - \
                0.5*np.linalg.norm(byz,axis=1)**2 + detJ

    def fit(self, x, kwargs_ae, kwargs_nf):
        """Trains the autoencoder and then the flow model"""
        # Autoencoder training
        self.history['ae'] = self._ae.fit(x=x, 
                                      y=x,  
                                      **kwargs_ae)

        # Encode training and validation data
        z = self.ae.encode(x)
        if 'validation_data' in kwargs_nf.keys():
            x_valid, y_valid = kwargs_nf['validation_data']
            kwargs_nf['validation_data'] = (self.ae.encode(x_valid), self.ae.encode(y_valid))
        
        #Train Normalizing Flow
        self.history['nf']= self._nf.fit(x=z,
                                         y=np.zeros(z.shape),
                                         **kwargs_nf)
        self.is_fit = True


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

    def compile_nf(self, loss=lambda _, log_p: -log_p):
        """Compile the Flow model using the given loss function"""
        if self._optim_nf is None:
            raise RuntimeError('Normalizing flow model not set. Use '
                               'builder.make_nf_model(cls, kwargs_dict)')
        self._nf.compile(self._optim_nf, loss)
        self._compiled_nf = True