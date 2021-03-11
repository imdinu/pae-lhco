import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.autoencoder import DenseAutoencoder
from models.flows import MAF
from models.nn import PaeBuilder

import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from loaders.LHCO import LhcoRnDLoader
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    print("Using Tensorflow:",tf.__version__)
    scaler = QuantileTransformer(output_distribution='uniform')
    files = {
        'bkg':'../data/3jet/BBOXbkg/bkgHLF_merged.h5',
        'bbox1_bkg':'../data/3jet/BBOX1/bkgHLF_merged.h5',
        'bbox1_sig':'../data/3jet/BBOX1/sigHLF_merged.h5'
    }

    train_fractions = {
        'bkg':'all'
    }

    test_fractions = {
        'bbox1_sig':'all',
        'bbox1_bkg':'all'
    }
    print("Loading data ...")
    loader = LhcoRnDLoader(files, 'all', scaler, exclude_range=[(2000,6000)])
    loader.preprocessing('bkg')
    train = loader.make_train_val(1_000_000, train_fractions, val_split=.2)
    test = loader.make_test(1_000_000, test_fractions, replace=False)

    print("x_test:",test['x_test'].shape)

    print("Computing mjj density ...")
    GMM = GaussianMixture
    gmm = GMM(n_components=200, covariance_type='full').fit(test["mjj_test"].reshape(-1, 1))
    plt.figure(figsize=(12,8))
    _, b, _ = plt.hist(train["mjj_train"], bins=50, label='mjj true', alpha=.5, density=True)
    sample = gmm.sample(train["mjj_train"].shape[0])
    plt.hist(sample[0], bins=b, label='mjj GMM', alpha=.5, density=True)
    plt.legend()
    plt.savefig("./figures/mjj_density.png")
    plt.close()#plt.show(block=False)

    weights2 = gmm.score_samples(train["mjj_train"].reshape(-1, 1))
    weights2_valid = gmm.score_samples(train["mjj_valid"].reshape(-1, 1))

    plt.figure(figsize=(12,8))
    plt.scatter(train["mjj_train"], 1/np.exp(weights2))
    plt.savefig("./figures/mjj_weights.png")
    plt.close()#plt.show(block=False)  

    tfd = tfp.distributions
    tfb = tfp.bijectors
    tfkl = tfk.layers

    print("Building models ...")
    builder = PaeBuilder()

    ae_config = {
        'input_dim':47, 
        'encoding_dim':10, 
        'units_list':[30],
        'weight_reg':tfk.regularizers.l1(1e-6),
        'output_activation':tf.nn.sigmoid
    }
    nf_config = {
        'n_dims':10, 
        'n_layers':5, 
        'units':[32 for i in range(4)]
    }
    optimizer_ae = {
        'lr': 0.05
    }
    optimizer_nf = {
        'lr': 0.005
    }

    builder.make_ae_model(DenseAutoencoder, ae_config)
    builder.make_nf_optimizer(tfk.optimizers.Adam, optimizer_ae)
    builder.make_nf_model(MAF, nf_config)
    builder.make_nf_optimizer(tfk.optimizers.Adam, optimizer_nf)
    builder.compile_ae()
    builder.compile_nf()
    pae = builder.pae

    ae_train = {
        'batch_size':200,
        'epochs':180,
        'sample_weight':1/np.exp(weights2),
        'validation_data':(train["x_valid"],train["x_valid"],1/np.exp(weights2_valid)),
        'callbacks':tfk.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=10,
            verbose=1
        )
    }

    nf_train = {
        'batch_size':200,
        'epochs':60,
        'validation_data':(train["x_valid"],train["x_valid"]),
        'callbacks':tfk.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=5,
            verbose=1
        )
    }
    with tf.device("/device:CPU:0"):
        pae.fit(train["x_train"],ae_train,nf_train)

    from utils.plotting import loss_plot, latent_space_plot, mjj_cut_plot, \
                           sculpting_plot, roc_plot

    loss_plot(pae.history, save_path="./figures/train.png")
    z_true = pae.ae.encode(train['x_train'])
    z_sample = pae.nf.sample(train['x_train'].shape[0])

    latent_space_plot(z_true, z_sample, save_path='figures/latent_space.png')

    mse = pae.reco_error(train['x_train'])
    pae.compute_implicit_sigma(train['x_valid'])
    ascore = -pae.anomaly_score(train['x_train'])

    mjj_cut_plot(mse, train['mjj_train'], prc=80, score_name='MSE', save_path='./figures/mse_cut_bkg.png')
    mjj_cut_plot(ascore, train['mjj_train'], prc=80, score_name='NLL', save_path='./figures/nll_cut_bkg.png')

    ano_scores = {
        'MSE': mse,
        'NLL': ascore
    }

    sculpting_plot(ano_scores, train['mjj_train'], max_prc=99, save_path='./figures/mass_sculpting_bkg.png')

    ascore_test = -pae.anomaly_score(test['x_test'])
    bkg, data = mjj_cut_plot(ascore_test, test['mjj_test'], prc=99, score_name='NLL', bins=100, save_path='./figures/cut_bbox1_samescaler.png')

    #key =  np.fromfile("../data/events_LHCO2020_BlackBox1.masterkey", sep='\n').astype(int)
    print("Ascore_test:",ascore_test.shape)
    test['mjj_test'].tofile('figures/mjj_test.npy')
    data.tofile('figures/data.npy')
    ascore_test.tofile('figures/ascore_test.npy')

    import pyBumpHunter as BH
    
    weights = np.repeat(1/(bkg.shape[0]/data.shape[0]),bkg.shape[0])
    hunter = BH.BumpHunter(rang=(3200,4800),
                        width_min=2,
                        width_max=6,
                        width_step=1,
                        scan_step=1,
                        Npe=10000,
                        Nworker=1,
                        seed=666,
                        weights=weights
                    )
    hunter.BumpScan(data,bkg)
    hunter.PlotBump(data,bkg,filename='./figures/bump_bbox1.png')
    hunter.PrintBumpTrue(data,bkg)
    