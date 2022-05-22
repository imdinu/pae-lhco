import os
import sys

import tensorflow_probability as tfp
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
tfd = tfp.distributions

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from scipy.spatial.distance import jensenshannon
import GPUtil

from .base import AbstractAnalysis
from pae.density.estimators import GMM, ConvKDE
from pae.models.autoencoder import DenseAutoencoder
from pae.models.flows import CondMAF, MAF
from pae.models.nn import PaeBuilder
from pae.loaders.LHCO import BaseDataloader
from pae.loaders import SCALERS
from pae.plotting import latent_space_plot, loss_plot
from pae.utils import load_json, dump_json

class HLFAnalysis(AbstractAnalysis):
    """Pae analysis workflow on LHCO high level features"""

    def __init__(self, config_path, dataset):
        """Creates a HLFAnalysis object.
        
        """
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True) 
            except RuntimeError as exception:
                # Memory growth must be set before GPUs have been initialized
                print(exception)

        self.STEPS = 500
        self.BATCH_SIZE = np.min([10_000, dataset['x_test'].shape[0]//10])
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(f"./testing/{self.timestamp}/")
        Path.mkdir(self.run_dir)
        config = load_json(config_path)
        config["NF:n_dims"] = config["AE:encoding_dim"]
        dump_json(config, self.run_dir/"CONFIG.json")
        self.estimator = config.pop("ANA:estimator")
        self.percentile = config.pop("ANA:percentile")
        self.n_folds = config.pop("ANA:folds")
        self.weight_key = config.pop("ANA:fit_key")
        self.sig_label = config.pop("ANA:sig_label")
        self.pae_config = config
        self.c_inputs = None
        self.dataset = dataset
        self.kfold = KFold(self.n_folds, shuffle=True)
        self.folds = self.kfold.split(dataset["x_train"])
        self.reweighting(fit_key=self.weight_key)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if isinstance(estimator, str):
            self._estimator = self.ESTIMATORS[estimator](
                **self.ESTIMATOR_KWARGS[estimator])
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

    def reweighting(self, estimator=None, fit_key=None, range=None):
        """Computes event weights based on mjj.

        Args:
            estimator: Instance of density estimator or string.
            fit_key: Dict key pointing to the dataset to be used for estimator 
                     training.
            range: Tuple containing the range of values (used for interpolation)
        """
        if estimator:
            self.estimator = estimator
        if fit_key:
            self.estimator.fit(self.dataset[fit_key], range=range)

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


    def cross_validate(self):
        """Trains the model and evaluates the performance for every fold using
        the `n_folds` parameter to determine the number of splits

        Returns:
            None
        """
        for i_fold, (i_train, i_valid) in enumerate(self.folds):
            self.pae, ae_train, nf_train = PaeBuilder.from_json(
                                                    self.pae_config.copy())
 
            ae_train['sample_weight'] = self.weights["train"][i_train]
            ae_train['verbose'] = 0
            ae_train['validation_data'] = (self.dataset["x_train"][i_valid],
                                                self.dataset["x_train"][i_valid], 
                                                self.weights["train"][i_valid])

            nf_train['verbose'] = 0
            nf_train['validation_data'] = (self.dataset["x_train"][i_valid],
                                            self.dataset["x_train"][i_valid])

            self.train(self.dataset["x_train"][i_train], ae_train, nf_train)
            self.plot_training(filename=self.run_dir/f"training{i_fold:02d}.pdf")
            self.plot_latent_space(filename=self.run_dir/f"latent{i_fold:02d}.pdf")
            self.pae.compute_implicit_sigma(self.dataset["x_train"][i_valid])
            results = self.evaluate(self.percentile, self.sig_label, js_bins=60)
            dump_json(results, self.run_dir/f"results{i_fold:02d}.json")
            results = self.plot_js_divergence(filename=self.run_dir/f"js_div{i_fold:02d}.pdf")
            dump_json(results, self.run_dir/f"js_div{i_fold:02d}.json")
            results = self.plot_roc(filename=self.run_dir/f"roc{i_fold:02d}.pdf")
            dump_json(results, self.run_dir/f"roc{i_fold:02d}.json")
            self._init_optimization()
            mse, nmse, lpz, ascore = self.pae_optimization(key="x_train")
            results = self.plot_js_divergence_opt(mse, nmse, lpz, ascore, 
                                filename=self.run_dir/f"js_div_opt{i_fold:02d}.pdf")
            dump_json(results, self.run_dir/f"js_div_opt{i_fold:02d}.json")
            mse, nmse, lpz, ascore = self.pae_optimization(key="x_test")
            results = self.plot_roc_opt(mse, nmse, lpz, ascore, 
                                filename=self.run_dir/f"roc_opt{i_fold:02d}.pdf")
            dump_json(results, self.run_dir/f"roc_opt{i_fold:02d}.json")


    def train(self, data, ae_train, nf_train, device=None):
        """Trains the pae model using the `_ae_train` and `_nf_train` 
        configuration dictionaries.
        
        Args:
            device : Name of the device to perform the training on
        """
        device = self._get_device() if not device else device
        with tf.device(device):
            self.pae.fit(data,
                        None,
                        ae_train,
                        nf_train)

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

        labels = np.array(list(map(self.binarize, self.dataset['labels_test'])))
        sig_label = (labels==1)
        bkg_label = (labels==0)

        # Compute signal efficiecy and background rejection
        sig_eff = sum(i_prc&sig_label)/sum(sig_label)
        bkg_rej = 1-sum(i_prc&bkg_label)/sum(bkg_label)

        # Compute AUC on the test set
        fpr, tpr, _ = roc_curve(labels, ascore_test)
        test_auc = auc(1-fpr, tpr)

        results = {
            'js_div_train': js_div_train,
            'js_div_test': js_div_test,
            'sig_eff': sig_eff,
            'bkg_rej': bkg_rej,
            'auc': test_auc
        }

        return results

    def mass_sculpting(self, mjj, score):
        max_prc = 99
        n_full, b = np.histogram(mjj, bins=60, density=True)
        js_div = {}
        for prc in range(1, max_prc+1):
            x_prc = np.percentile(score, prc)
            i_prc = np.where(score >= x_prc)[0]
            n_prc, _ = np.histogram(mjj[i_prc], bins=b, density=True)
            js_div[prc] = jensenshannon(n_full,n_prc)

        return js_div

    def nmse(self, x, pae):
        reco_error = np.square(pae.ae(x)-x)
        return np.dot(reco_error,pae.sigma_square**(-1))

    def plot_training(self, filename=None):
        fig = loss_plot(self.pae.history)
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')


    def plot_latent_space(self, filename=None):
        z_true = self.pae.ae.encode(self.dataset['x_train'])

        if self.c_inputs is not None:
            c = self.c_inputs['train'] 
            z_sample = self.pae.nf.sample(c, self.dataset['x_train'].shape[0])
        else:
             z_sample = self.pae.nf.sample(self.dataset['x_train'].shape[0])

        fig = latent_space_plot(z_true, z_sample)
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')

    def plot_js_divergence(self, filename=None):
        mjj = self.dataset['mjj_train']

        score = -self.pae.anomaly_score(self.dataset['x_train'])
        js_div_pae = self.mass_sculpting(mjj, score)

        score = self.nmse(self.dataset['x_train'], self.pae)
        js_div_nmse = self.mass_sculpting(mjj,score)

        score = self.pae.reco_error(self.dataset['x_train'])
        js_div_mse = self.mass_sculpting(mjj,score)

        score = -self.pae.log_prob_encoding(self.dataset['x_train'])
        js_div_lpz = self.mass_sculpting(mjj,score)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_lpz.values()), mode='lines',
                name=r"$-\log p_z$", line=dict(color="chocolate", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_mse.values()), mode='lines',
                name=r"$\text{MSE}$", line=dict(color="steelblue", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_nmse.values()), mode='lines',
                name=r"$\text{MSE} \cdot \sigma^{\circ-2}$", line=dict(color="cornflowerblue", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_pae.values()), mode='lines',
                name=r"$\text{PAE}$", line=dict(color="plum", width=3))
        )

        fig.update_layout(
            title_text = "Mass sculpting",
            xaxis_title = "Percentile Cut",
            yaxis_title = "Jensen–Shannon",
            margin={'l': 80, 'b': 40, 't': 40, 'r': 0},
            width=600, height=500,
            paper_bgcolor='rgba(0,0,0,1)',
                legend = dict(x=0, y=1,
                traceorder='normal',
                font=dict(size=15))
        )
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')

        return {"cut": list(js_div_pae.keys()), "MSE": list(js_div_mse.values()),
                "NMSE": list(js_div_nmse.values()), "logPz": list(js_div_lpz.values()),
                "PAE": list(js_div_pae.values())}

    def plot_js_divergence_opt(self, mse, nmse, lpz, ascore, filename=None):
        mjj = self.dataset['mjj_train']
        js_div_pae = self.mass_sculpting(mjj, ascore)
        js_div_nmse = self.mass_sculpting(mjj, nmse)
        js_div_mse = self.mass_sculpting(mjj, mse)
        js_div_lpz = self.mass_sculpting(mjj, lpz)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_lpz.values()), mode='lines',
                name=r"$-\log p_z$", line=dict(color="chocolate", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_mse.values()), mode='lines',
                name=r"$\text{MSE}$", line=dict(color="steelblue", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_nmse.values()), mode='lines',
                name=r"$\text{MSE} \cdot \sigma^{\circ-2}$", line=dict(color="cornflowerblue", width=3))
        )
        fig.add_trace(
            go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_pae.values()), mode='lines',
                name=r"$\text{PAE}$", line=dict(color="plum", width=3))
        )

        fig.update_layout(
            title_text = "Mass sculpting",
            xaxis_title = "Percentile Cut",
            yaxis_title = "Jensen–Shannon",
            margin={'l': 80, 'b': 40, 't': 40, 'r': 0},
            width=600, height=500,
            paper_bgcolor='rgba(0,0,0,1)',
                legend = dict(x=0, y=1,
                traceorder='normal',
                font=dict(size=15))
        )
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')

        return {"cut": list(js_div_pae.keys()), "MSE": list(js_div_mse.values()),
                "NMSE": list(js_div_nmse.values()), "logPz": list(js_div_lpz.values()),
                "PAE": list(js_div_pae.values())}

    def plot_roc(self, filename):
        labels = np.array(list(map(self.binarize, self.dataset['labels_test'])))
        results = {}

        score = self.pae.reco_error(self.dataset['x_test'])
        roc_mse = self._make_roc_trace(labels, score, 'steelblue', "MSE", results)

        score = self.nmse(self.dataset['x_test'], self.pae)
        roc_nmse = self._make_roc_trace(labels, score, 'cornflowerblue', "NMSE", results)

        score = -self.pae.log_prob_encoding(self.dataset['x_test'])
        roc_lpz = self._make_roc_trace(labels, score, 'chocolate', "logPz", results)

        score = -self.pae.anomaly_score(self.dataset['x_test'])
        roc_pae = self._make_roc_trace(labels, score, 'plum', "PAE", results)

        fig = go.Figure()
        fig.add_trace(roc_mse)
        fig.add_trace(roc_nmse)
        fig.add_trace(roc_lpz)
        fig.add_trace(roc_pae)
        fig.update_layout(
            width=500, height=500,
            xaxis_title = "Signal efficiency",
            yaxis_title = "Background Rejection",
            margin={'l': 60, 'b': 60, 't': 40, 'r': 0},
            legend = dict(x=0.1, y=0.05,
                traceorder='normal',
                font=dict(size=15)),
            title_text="ROC curves",
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,1)',
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')
        return results

    def plot_roc_opt(self, mse, nmse, lpz, ascore, filename=None):
        labels = np.array(list(map(self.binarize, self.dataset['labels_test'])))

        results = {}
        roc_mse = self._make_roc_trace(labels, mse, 'steelblue', "MSE", results)
        roc_nmse = self._make_roc_trace(labels, nmse, 'cornflowerblue', "NMSE", results)
        roc_lpz = self._make_roc_trace(labels, lpz, 'chocolate', "logPz", results)
        roc_pae = self._make_roc_trace(labels, ascore, 'plum', "PAE", results)

        fig = go.Figure()
        fig.add_trace(roc_mse)
        fig.add_trace(roc_nmse)
        fig.add_trace(roc_lpz)
        fig.add_trace(roc_pae)
        fig.update_layout(
            width=500, height=500,
            xaxis_title = "Signal efficiency",
            yaxis_title = "Background Rejection",
            margin={'l': 60, 'b': 60, 't': 40, 'r': 0},
            legend = dict(x=0.1, y=0.05,
                traceorder='normal',
                font=dict(size=15)),
            title_text="ROC curves",
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,1)',
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        if filename:
            fig.write_image(filename)
        else:
            fig.show('vscode')

        return results

    def _make_roc_trace(self, labels, score, c, n="", d=None):
        fpr, tpr, _ = roc_curve(labels, score)
        aauc = auc(1-fpr, tpr)
        if d is not None and n != "":
            d[f"{n}_sig_eff"] = tpr.tolist()
            d[f"{n}_bkg_rej"] = (1-fpr).tolist()
        return go.Scatter(x=tpr, y=1-fpr, mode='lines',
            name=n+f"AUC:{aauc:.2f}", line=dict(color=c, width=2))

    def binarize(self, label):
        return 1 if label == self.sig_label else 0

    def _get_device(self, prioritize_gpu=True):
        if prioritize_gpu:
            gpus = tf.config.experimental.list_logical_devices('GPU')
            gpu_no = GPUtil.getFirstAvailable(order = 'load', 
                            maxLoad=0.5, maxMemory=0.5, 
                            attempts=1, interval=900)[0]
            device = gpus[gpu_no]
        else:        
            cpus = tf.config.experimental.list_logical_devices('CPU')
            device = cpus[0]
        return device.name

    def _init_optimization(self):
        self.sigma = tf.constant(tf.sqrt(self.pae.sigma_square))
        self.z_ = tf.Variable(self.pae.ae.encoder(
                self.dataset['x_test'][:self.BATCH_SIZE].astype(np.float32)))
        self.opt = tf.optimizers.Adam(learning_rate=0.001)

    def pae_optimization(self, key="x_test"):
        ds_ = tf.convert_to_tensor(self.dataset[key], dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices(ds_)
        ds = ds.cache()
        ds = ds.batch(self.BATCH_SIZE)
        ds = ds.prefetch(self.BATCH_SIZE)


        with tf.device(self._get_device()):
            ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
            for i, batch in enumerate(ds):
                ta = ta.write(i, self.find_map(batch))
            z_map = ta.concat()
            del ta

        with tf.device(self._get_device()):
            byz = self.pae.nf.inverse(z_map)
            detJ = self.pae.nf.inverse_log_det_jacobian(z_map)
            x = self.pae.ae.decode(z_map)
            reco_error = np.square(x-self.dataset[key])
            ascore = +0.5*np.dot(reco_error,self.pae.sigma_square**(-1)) + \
                    0.5*np.linalg.norm(byz,axis=1)**2 - detJ
            lp = -self.pae.nf(z_map)
            mse = np.mean(reco_error, axis=1)
            mses = np.dot(reco_error,self.pae.sigma_square**(-1))

        return mse, mses, lp, ascore

    @tf.function
    def max_apriori_prob(self, x, z, sigma, pae):
        distrs = tfd.MultivariateNormalDiag(loc=x, scale_diag=sigma)
        nf_ll = pae.nf(z)
        reco = pae.ae.decoder(z)
        gauss_ll = distrs.log_prob(reco)
        return  tf.reduce_mean(-nf_ll - gauss_ll) 


    @tf.function
    def find_map(self, x_):
        if self.z_ is None:
            self.z_ = tf.Variable(self.pae.ae.encoder(x_))
        self.z_.assign(self.pae.ae.encoder(x_))
        for _ in range(self.STEPS):
            with tf.GradientTape() as tape:
                tape.watch(self.z_)
                nll = self.max_apriori_prob(x_, self.z_, self.sigma, self.pae)
            grad = tape.gradient(nll, [self.z_])
            self.opt.apply_gradients(zip(grad, [self.z_]))
        return self.z_



        

