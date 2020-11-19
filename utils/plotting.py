import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

sns.set_context('talk')

def loss_plot(history, save_path: str = None):
    history_ae = history['ae'].history
    history_nf = history['nf'].history
    epochs_ae = history['ae'].epoch
    epochs_nf = history['nf'].epoch

    fig = plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.plot(history_ae['loss'], label="Training Loss" , zorder=1)
    plt.scatter(epochs_ae, history_ae['val_loss'], label="Validation Loss", 
                    color="orange", zorder=2)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.title("Autoencoder loss")

    plt.subplot(122)
    plt.plot(history_nf['loss'], label="Training Loss" , zorder=1)
    plt.scatter(epochs_nf, history_nf['val_loss'], label="Validation Loss", 
                    color="orange", zorder=2)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.title("Normalizing Flow loss")
    if save_path is not None:
        plt.savefig(save_path)
    fig.show()

def latent_space_plot(pae, x_test, 
                      bins: int = 20, 
                      save_path: str = None):
    z_true = pae.ae.encode(x_test)
    z_sample = pae.nf.sample(x_test.shape[0])

    r, c = optimal_grid(z_true.shape[1])
    fig = plt.figure(figsize=(15,8))
    for i in range(z_true.shape[1]):
        plt.subplot(r, c, i+1)
        n1, b, _ = plt.hist(z_true[:,i], bins=bins, density=True, alpha=0.5)
        n2, _, _ = plt.hist(z_sample[:,i], bins=b, density=True, alpha=0.5)
        plt.tight_layout()
        plt.title(f"kl={entropy(n1, n2):.6f}")
    
    blue_patch = mpatches.Patch(color='steelblue', label='The red data', 
                                alpha = 0.5)
    orange_patch = mpatches.Patch(color='orange', label='The red data', 
                                  alpha = 0.5)
    plt.figlegend((blue_patch, orange_patch), 
                  ("Reconstructed", "Sampled from NF"), 
                  'lower right', fancybox=True, framealpha=1.)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def mjj_cut_plot(ano_score, mjj, 
                 prc: int = 75, 
                 bins: int = 60, 
                 score_name: str = 'anomaly score', 
                 save_path: str = None):
    
    x_min = np.percentile(ano_score, 0.5)
    x_max = np.percentile(ano_score, 99.5)
    x_prc = np.percentile(ano_score, prc)
    i_prc = np.where(ano_score >= x_prc)[0]

    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.hist(ano_score, bins=bins, density=True, alpha=.8, 
             label='Test dataset')
    plt.xlim(x_min,x_max)
    plt.axvline(x_prc, color='red', label=f'{prc}''$^{th}$ percentile')
    plt.legend()
    plt.xlabel(f'{score_name}')

    plt.subplot(1,2,2)
    _, b, _ = plt.hist(mjj, bins=60, density=True, alpha=.5, 
                       label='Full test datset')
    _, _, _ = plt.hist(mjj[i_prc], bins=b, density=True, alpha=.5, 
                       label=f'{prc}''$^{th}$'f' {score_name} percentile+')
    plt.xlabel('$m_{jj}$')
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def sculpting_plot(ano_score, mjj, 
                   max_prc: int = 90, 
                   bins: int = 60,
                   save_path: str = None):

    n_full, b = np.histogram(mjj, bins=bins, density=True)
    js_divs = {}
    for prc in range(1, max_prc+1):
        x_prc = np.percentile(ano_score, prc)
        i_prc = np.where(ano_score >= x_prc)[0]
        n_prc, _ = np.histogram(mjj[i_prc], bins=b, density=True)
        js_divs[prc] = jensenshannon(n_full,n_prc)
    plt.figure(figsize=(10,6))
    plt.plot(js_divs.keys(), js_divs.values(), label='Jensen-Shannon divergence')
    plt.ylabel('Mass sculpting')
    plt.xlabel('Percentile cut')
    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def optimal_grid(n):
    rows = np.floor(np.sqrt(n))
    residual = 1 if n%rows != 0 else 0
    cols = n//rows + residual
    return int(rows), int(cols)