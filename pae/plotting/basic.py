import numpy as np
from scipy.stats import entropy

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.subplots import make_subplots

from loaders import FEATURE_SETS
from .utils import optimal_grid

def feature_plots(data, features='all', color='steelblue'):
    """Return a figure of all histogramsof the columns in the input data. 
    """
    if isinstance(features, str):
        features = FEATURE_SETS[features]

    cols = data.shape[1]//4
    rows = data.shape[1]//cols +1

    fig = make_subplots(rows=rows, cols=cols)
    for i in range(data.shape[1]):
        counts, bins = np.histogram(data[:,i], bins=20)
        fig.add_trace(
            go.Bar(x=bins, y=counts, 
                   name=features[i], marker_color=color),
            row=i//cols+1, col=i%cols+1
        )
    fig.update_layout(bargap=0,
                    height=400, 
                    width=900,
                    font=dict(size=8),
                    showlegend=False)
    return fig

def loss_plot(history):
    history_ae = history['ae'].history
    history_nf = history['nf'].history
    epochs_ae = history['ae'].epoch
    epochs_nf = history['nf'].epoch
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=epochs_ae, y=history_ae['loss'], 
                            mode='lines', marker_color='steelblue', name='AE: loss'),
                            row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_ae, y=history_ae['val_loss'], 
                            mode='markers', marker_color='darkorange', name='AE: val_loss'),
                            row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_nf, y=history_nf['loss'],
                            mode='lines', marker_color='steelblue', name='NF: loss'),
                            row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs_nf, y=history_nf['val_loss'], 
                            mode='markers', marker_color='darkorange', name='NF: val_loss'),
                            row=1, col=2)
    fig.update_layout(title="Training and validation loss", width=800)
    return fig

def latent_space_plot(z_true, z_sample,
                      bins: int = 20
                      ):

    rows, cols = optimal_grid(z_true.shape[1])
    data = {'true':[], 'sample':[], 'bins':[], 'kl':[]}
    for i in range(z_true.shape[1]):
        n1, b = np.histogram(z_true[:,i], bins=bins, density=True)
        n2, _ = np.histogram(z_sample[:,i], bins=b, density=True)
        kl=f"kl={entropy(n1, n2):.2e}"
        data['bins'].append(b[:-1])
        data['true'].append(n1)
        data['sample'].append(n2)
        data['kl'].append(kl)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=data['kl'])
    for i in range(z_true.shape[1]):
        fig.add_trace(go.Bar(x=data['bins'][i], y=data['true'][i],
                name='True latent', marker_color='steelblue',
                showlegend=(i==0)),
            row=i//cols+1, col=i%cols+1
            )
        fig.add_trace(go.Bar(x=data['bins'][i], y=data['sample'][i],
                name='Sample from NF', marker_color='coral',
                showlegend=(i==0)),
            row=i//cols+1, col=i%cols+1
            )
    fig.update_layout(
            bargap=0,
            height=400, 
            width=900,
            title_text='Latent space plot')
    return fig