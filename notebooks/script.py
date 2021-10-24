#!/usr/bin/env python
import sys
import os
import pathlib
from datetime import datetime

import GPUtil
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow_addons as tfa
from sklearn.model_selection import KFold

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

sys.path.append("../")

from pae.models.autoencoder import DenseAutoencoder
from pae.models.flows import MAF
from pae.models.nn import PaeBuilder

from pae.utils import load_json
from pae.loaders.LHCO import ScalarLoaderLHCO, DatasetBuilder

from pae.plotting import feature_plots, loss_plot, latent_space_plot

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
#os.environ['TF_GPU_THREAD_COUNT'] = '1'

ds_options = tf.data.Options()
ds_options.experimental_optimization.map_parallelization = True
ds_options.experimental_optimization.parallel_batch = True

devices = tf.config.list_physical_devices()
print("tensorflow", tf.__version__)
print("tensorflow-probability", tfp.__version__)
print("Available devices:", *[dev[1] for dev in devices])

# SEED = 100
# np.random.seed(SEED) 
# tf.random.set_seed(SEED)

# In[1]:

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = pathlib.Path(f"./logs/{timestamp}")
os.mkdir(run_dir)

# In[ ]:

# Load Datasets
x = ScalarLoaderLHCO.from_json("../pae/configs/loader/rnd_scalar_2j.json")
mjj = ScalarLoaderLHCO.from_json("../pae/configs/loader/rnd_scalar_mjj.json")
builder = DatasetBuilder(x, mjj)
builder.data_preparation(sample_sizes ={'sig':100_000, 'bkg': 1_000_000}, fit_key='bkg')
dataset = builder.make_dataset(train = {'bkg':900_000}, test={'sig':100_000, 'bkg': 100_000})
#dataset = builder.make_dataset(train = {'bkg':9000}, test={'sig':100, 'bkg': 9900})

# Plot features
fig = feature_plots(dataset['x_train'], 'dijet')
fig.update_layout(title="Training features transformed")
fig.write_image(run_dir / "train_features.pdf")
fig = feature_plots(dataset['x_test'], 'dijet', color='coral')
fig.update_layout(title="Testing features transformed")
fig.write_image(run_dir / "test_features.pdf")



from pae.density import GMM, ConvKDE, KNNDensity, ExpnormFit

density_estimator = ConvKDE(bw=50, kernel="box")
fit_data = dataset["mjj_train"]

density_estimator.fit(fit_data)

x_ref = np.linspace(1600, 8000, 1701)
y_kde = density_estimator.evaluate(x_ref)

fig = go.Figure()
# fig.add_trace(go.Scatter(x=x_ref, y=y_gmm, mode='lines', name='GMM',
#                          line={'color': 'greenyellow', 'width': 2, 'dash': 'dot'}))
fig.add_trace(go.Scatter(x=x_ref, y=y_kde, mode='lines', name='FFTKDE',
                         line={'color': 'indianred', 'width': 2, 'dash': 'dash'}))
# fig.add_trace(go.Scatter(x=x_ref, y=y_knn, mode='lines', name='KNN',
#                          line={'color': 'turquoise', 'width': 2, 'dash': 'longdashdot'}))
# fig.add_trace(go.Scatter(x=x_ref, y=y_exp, mode='lines', name='expnorm',
#                          line={'color': 'indigo', 'width': 2, 'dash': 'solid'}))
fig.add_trace(go.Histogram(x=dataset["mjj_train"].ravel(), nbinsx=600, histnorm='probability density', 
                           marker_color='steelblue', name='Histnorm'))
fig.update_layout(
    title_text='Dijet mass distribution and density estimation',
    xaxis_title_text=r'$$m_{jj}$$',
    yaxis_title_text=r'density',
)
fig.write_image(run_dir / "density_fit.pdf")


# In[ ]:

mjj_train = dataset['mjj_train']
data_key = 'mjj_train'

# w_gmm = gmm.get_weights(dataset[data_key])
w_kde = density_estimator.get_weights(mjj_train.ravel())
# w_expnorm = expn.get_weights(dataset[data_key])
# w_knn = knn.get_weights(dataset[data_key])
fig = go.Figure()
# fig.add_trace(go.Scattergl(x=data, y=w_gmm, 
#                            mode='markers', name='GMM', opacity=0.8,
#                            marker=dict(color='greenyellow',symbol='diamond'))
#             )
fig.add_trace(go.Scattergl(x=mjj_train, y=w_kde, 
                           mode='markers', name='FFTKDE', opacity=0.8,
                           marker=dict(color='indianred',symbol='star-square'))
            )
# fig.add_trace(go.Scattergl(x=data, y=w_expnorm, 
#                            mode='markers', name='expnorm', opacity=0.8,
#                            marker=dict(color='indigo',symbol='circle'))
#             )
# fig.add_trace(go.Scattergl(x=data, y=w_knn, 
#                            mode='markers', name='KNN', opacity=0.8,
#                            marker=dict(color='turquoise',symbol='triangle-nw-dot'))
#             )
fig.update_layout(
    title_text='Weights relative to dijetmass scatter plot',
    xaxis_title_text=r'$$m_{jj}$$',
    yaxis_title_text=r'weight',
    yaxis_type="log"
)
fig.write_image(run_dir / "weights_scatter.pdf")


# In[ ]:

# n_gmm, b = np.histogram(data, bins=20, weights=w_gmm)
n_kde, b = np.histogram(mjj_train.ravel(), bins=20, weights=w_kde)
# n_exp, _ = np.histogram(data, bins=b, weights=w_expnorm)
# n_knn, _ = np.histogram(data, bins=b, weights=w_knn)
fig = go.Figure()       
# fig.add_trace(go.Bar(x=b[:-1], y=n_gmm, name='GMM',
#                         marker=dict(color='yellowgreen'))
#             )
fig.add_trace(go.Bar(x=b[:-1], y=n_kde, name='FFTKDE',
                           marker=dict(color='indianred'))
            )
# fig.add_trace(go.Bar(x=b[:-1], y=n_exp, name='expnorm',
#                         marker=dict(color='indigo'))
#             )
# fig.add_trace(go.Bar(x=b[:-1], y=n_knn, name='KNN',
#                         marker=dict(color='turquoise'))
#             )
fig.update_layout(
    title_text=r'Weighted dijet mass bins',
    xaxis_title_text=r'$$m_{jj}$$',
    yaxis_title_text=r'Counts',
    yaxis_type="log",
    bargap=0.1
)
fig.write_image(run_dir / "reweighted_mass.pdf")



fold5 = KFold(8, shuffle=True)
q= fold5.split(dataset["x_train"])
x_train, x_valid = next(q)
print(x_train.shape)
print(x_valid.shape)


builder = PaeBuilder()
ae_config = {
    'input_dim':47, 
    'encoding_dim':10, 
    'units':[30, 20, 15],
    'weight_reg':tfk.regularizers.L1L2(l1=1e-5, l2=1e-4),
    'output_activation':tf.nn.sigmoid
}
nf_config = {
    'n_dims':10, 
    'n_layers':5, 
    'units':[32 for _ in range(4)]
}
optimizer_ae = {
    'learning_rate': 0.001
}
optimizer_nf = {
    'learning_rate': 0.005
}

builder.make_ae_model(DenseAutoencoder, ae_config)
builder.make_ae_optimizer(tfk.optimizers.Adam, optimizer_ae)
builder.make_nf_model(MAF, nf_config)
builder.make_nf_optimizer(tfk.optimizers.Adam, optimizer_nf)
builder.compile_ae()
builder.compile_nf()
pae = builder.pae

# make data pipeline
def make_dataset_ae(x, w):
    x_ds = tf.data.Dataset.from_tensor_slices(x)
    w_ds = tf.data.Dataset.from_tensor_slices(w)
    return tf.data.Dataset.zip((x_ds, x_ds, w_ds))

weights = w_kde

tensorboard_callback_0 = tf.keras.callbacks.TensorBoard(log_dir=run_dir / "ae", 
                        histogram_freq=1, write_images=True, profile_batch=(2,10),
                        update_freq="batch")
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=run_dir / "nf", 
                        histogram_freq=1, write_images=True, profile_batch=(2,10),
                        update_freq="batch")

ds_train = make_dataset_ae(dataset["x_train"][x_train], weights[x_train]).with_options(ds_options)
ds_valid = make_dataset_ae(dataset["x_train"][x_valid], weights[x_valid]).with_options(ds_options)

ds_train = ds_train.cache()
ds_train = ds_train.batch(200)
ds_train = ds_train.prefetch(200)
ds_valid = ds_valid.cache()
ds_valid = ds_valid.batch(200)
ds_valid = ds_valid.prefetch(200)


tqdm_callback = tfa.callbacks.TQDMProgressBar()
# Training Configuration
ae_train ={
    'epochs':120,
    'validation_data':ds_valid,
    'callbacks':[tfk.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=10,
        verbose=1
    ), tqdm_callback, tensorboard_callback_0],
    "verbose":0
}
nf_train ={
    'epochs':80,
    'callbacks':[tfk.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=5,
        verbose=1
    ), tqdm_callback, tensorboard_callback_1],
    "verbose":0
}

device_id = GPUtil.getFirstAvailable(order = 'load', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=True)
if not device_id:
    raise RuntimeError("No GPU Available")
with tf.device(f"/device:GPU:{device_id[0]}"):
    pae.fit_ae(ds_train, **ae_train)
with tf.device("/device:CPU:0"):
    z = pae.ae.encode(dataset["x_train"][x_train])
    z_valid = pae.ae.encode(dataset["x_train"][x_valid])

    ds_train = tf.data.Dataset.from_tensor_slices((
        z, np.zeros(z.shape))).with_options(ds_options)
  
    ds_valid = tf.data.Dataset.from_tensor_slices((
        z_valid, np.zeros(z_valid.shape))).with_options(ds_options)

    ds_train = ds_train.cache()
    ds_train = ds_train.batch(200)
    ds_train = ds_train.prefetch(200)
    ds_valid = ds_valid.cache()
    ds_valid = ds_valid.batch(200)
    ds_valid = ds_valid.prefetch(200)

    nf_train["validation_data"] = ds_valid
    pae.fit_nf(ds_train, **nf_train)

# SAVE MODELS
pae.ae.save_weights(run_dir / "ae.h5")
pae.nf.save_weights(run_dir /"nf.h5")

# PLOT LOSS AND LATENT SPACE
fig = loss_plot(pae.history)
fig.write_image(run_dir / "loss_plot.pdf")

z_true = pae.ae.encode(dataset['x_train'])
z_sample = pae.nf.sample(dataset['x_train'].shape[0])

fig = latent_space_plot(z_true, z_sample)
fig.write_image(run_dir / "latent_space.pdf")

# CALCULATE ANOMALY SCORES
pae.compute_implicit_sigma(dataset['x_train'][x_valid])
ascore = -pae.anomaly_score(dataset['x_test'])
mse = pae.reco_error(dataset['x_test'])
x=dataset['x_test']
mses = np.dot(np.square(pae.ae(x)-x),pae.sigma_square**(-1))
lp = np.exp(np.array(pae.log_prob_encoding(x)))


# CLCULATE TEST MAP
tfd = tfp.distributions

sigma = tf.constant(tf.sqrt(pae.sigma_square))
BATCH_SIZE_MAP = np.min([50_000, dataset['x_test'].shape[0]//10])
z_ = tf.Variable(pae.ae.encoder(dataset['x_test'][:BATCH_SIZE_MAP].astype(np.float32))) #tf.Variable(pae.ae.encoder(dataset['x_test']))
opt = tf.optimizers.Adam(learning_rate=0.001)
STEPS=500
map_summary_writer = tf.summary.create_file_writer(str(run_dir / "map"))


@tf.function
def max_apriori_prob(x, z, sigma, pae):
    distrs = tfd.MultivariateNormalDiag(loc=x, scale_diag=sigma)
    nf_ll = pae.nf(z)
    reco = pae.ae.decoder(z)
    gauss_ll = distrs.log_prob(reco)
    return  tf.reduce_mean(-nf_ll - gauss_ll) 


@tf.function
def find_map(x_):
    global z_
    if z_ is None:
        z_ = tf.Variable(pae.ae.encoder(x_))
    z_.assign(pae.ae.encoder(x_))
    for i in range(STEPS):
        with tf.GradientTape() as tape:
            tape.watch(z_)
            nll = max_apriori_prob(x_, z_, sigma, pae)
        grad = tape.gradient(nll, [z_])
        opt.apply_gradients(zip(grad, [z_]))
        with map_summary_writer.as_default():
            tf.summary.scalar('nll', nll, step=i)
    return z_

ds_ = tf.convert_to_tensor(dataset['x_test'], dtype=tf.float32)
ds = tf.data.Dataset.from_tensor_slices(ds_)
ds = ds.cache()
ds = ds.batch(BATCH_SIZE_MAP)
ds = ds.prefetch(BATCH_SIZE_MAP)

tf.profiler.experimental.start(str(run_dir / "map"))
with tf.device(f"/device:GPU:{device_id[0]}"):
#with tf.device("/device:CPU:0"):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
    for i, batch in enumerate(ds):
        ta = ta.write(i, find_map(batch))
    z_map = ta.concat()
    del ta
tf.profiler.experimental.stop()

with tf.device(f"/device:GPU:{device_id[0]}"):
    byz = pae.nf.inverse(z_map)
    detJ = pae.nf.inverse_log_det_jacobian(z_map)
    x = pae.ae.decode(z_map)
    reco_error = np.square(x-dataset['x_test'])
    ascore2 = +0.5*np.dot(reco_error,pae.sigma_square**(-1)) + \
            0.5*np.linalg.norm(byz,axis=1)**2 - detJ
    lp2 = -pae.nf(z_map)
    mse2 = np.mean(reco_error, axis=1)
    mses2 = np.dot(reco_error,pae.sigma_square**(-1))

# ANOMALY SCORE PLOTS
prc=90

x_min = np.percentile(ascore, 1)
x_max = np.percentile(ascore, 99)
x_prc = np.percentile(ascore, prc)
i_prc = (ascore >= x_prc)

fig = go.Figure(layout_xaxis_range=[-30,30])
fig.add_trace(go.Histogram(x=ascore, name='Test dataset',
                           marker_color='plum', nbinsx=200),
              )
fig.add_vline(x=x_prc, y1=5100, line_width=2, line_color='firebrick', 
              annotation_text=f"{prc}th percentile", 
              annotation_position="top right",
              )

fig.update_layout(
    xaxis_title='Anomaly Score',
    #title_text=r'Cut on Anomaly Score',
    margin={'l': 80, 'b': 40, 't': 40, 'r': 40},
    width=600, height=300,
    paper_bgcolor='rgba(0,0,0,1)',
        font=dict(size=18))
fig.write_image(run_dir / "ascore.pdf")

# %%
x_min2 = np.percentile(ascore2, 1)
x_max2 = np.percentile(ascore2, 99)
x_prc2 = np.percentile(ascore2, prc)
i_prc2 = (ascore2 >= x_prc2)
fig = go.Figure()

fig.add_trace(go.Histogram(x=ascore2, name='Test dataset',
                           marker_color='plum', nbinsx=200),
              )
fig.add_vline(x=x_prc2, y1=5100, line_width=2, line_color='firebrick', 
              annotation_text=f"{prc}th percentile", 
              annotation_position="top right",
              )

fig.update_layout(
    xaxis_title='Anomaly Score',
    #title_text=r'Cut on Anomaly Score',
    margin={'l': 80, 'b': 40, 't': 40, 'r': 40},
    width=600, height=300,
    paper_bgcolor='rgba(0,0,0,1)',
        font=dict(size=18))
fig.write_image(run_dir / "ascore2.pdf")

# In[ ]:


def adj(x, prc_min=1, prc_max=99):
    xmin, xmax = np.percentile(x,prc_min), np.percentile(x,prc_max)
    return x[(x >= xmin) & (x<= xmax)]

mjj=dataset['mjj_test']


def binarize(label):
    return 1 if label == 'sig' else 0
labels = np.array(list(map(binarize, dataset['labels_test'])))
sig_label = (labels==1)
bkg_label = (labels==0)

fig = go.Figure()
fig.add_trace(go.Histogram(x=adj(mjj[bkg_label].ravel()), name="SM - QCD",
                          marker_color='steelblue', nbinsx=150))
fig.add_trace(go.Histogram(x=mjj[sig_label][:2000].ravel(), name="BSM - Signal",
                          marker_color='darkorange'))
sb = 100*sum(sig_label)/sum(bkg_label)
fig.update_layout(
    xaxis_title=r'$m_{jj}$',
    title_text='Dijet mass spectrum',
    barmode='stack',
    legend=dict(x=0.78, y=1, traceorder='normal', font=dict(size=15)),
    paper_bgcolor='rgba(0,0,0,1)',
    width=800,
    height=500,
)

fig.write_image(run_dir / "pre_cut.pdf") 

fig = go.Figure()
fig.add_trace(go.Histogram(x=mjj[i_prc&bkg_label].ravel(), name="Full test bkg",
                          marker_color='steelblue', nbinsx=100))
fig.add_trace(go.Histogram(x=mjj[i_prc&sig_label].ravel(), name="Full test sig",
                          marker_color='coral'))
sb = 100*sum(i_prc&sig_label)/sum(i_prc&bkg_label)
fig.update_layout(
    xaxis_title='$m_{jj}$',
    title_text=f'Dijet mass spectrum after cut S/B={sb:.2f}%',
    width=600,
    barmode='stack'
    )
fig.write_image(run_dir / "post_cut.pdf") 

# %%
fig = go.Figure()
fig.add_trace(go.Histogram(x=mjj[i_prc2&bkg_label].ravel(), name="Full test bkg",
                          marker_color='steelblue', nbinsx=100))
fig.add_trace(go.Histogram(x=mjj[i_prc2&sig_label].ravel(), name="Full test sig",
                          marker_color='coral'))
sb = 100*sum(i_prc2&sig_label)/sum(i_prc2&bkg_label)
fig.update_layout(
    xaxis_title='$m_{jj}$',
    title_text=f'Dijet mass spectrum after cut S/B={sb:.2f}%',
    width=600,
    barmode='stack'
    )
fig.write_image(run_dir / "post_cut2.pdf") 

# In[ ]:


from scipy.spatial.distance import jensenshannon

def mass_sculpting(mjj, score):
    max_prc = 99
    n_full, b = np.histogram(mjj, bins=60, density=True)
    js_div = {}
    for prc in range(1, max_prc+1):
        x_prc = np.percentile(score, prc)
        i_prc = np.where(score >= x_prc)[0]
        n_prc, _ = np.histogram(mjj[i_prc], bins=b, density=True)
        js_div[prc] = jensenshannon(n_full,n_prc)

    return js_div

def nmse(x, pae):
    reco_error = np.square(pae.ae(x)-x)
    return np.dot(reco_error,pae.sigma_square**(-1))

pio.templates.default = "plotly_dark"
mjj = dataset['mjj_train']

score = -pae.anomaly_score(dataset['x_train'])
js_div_pae = mass_sculpting(mjj,score)

score = nmse(dataset['x_train'], pae)
js_div_nmse = mass_sculpting(mjj,score)


########################################
ds_2 = tf.convert_to_tensor(dataset['x_train'], dtype=tf.float32)
ds = tf.data.Dataset.from_tensor_slices(ds_2)
#ds = ds.cache()
ds = ds.batch(BATCH_SIZE_MAP)
ds = ds.prefetch(BATCH_SIZE_MAP)

tf.profiler.experimental.start(str(run_dir / "map2"))
with tf.device(f"/device:GPU:{device_id[0]}"):
#with tf.device("/device:CPU:0"):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
    for i, batch in enumerate(ds):
        ta = ta.write(i, find_map(batch))
    z_map_train = ta.concat()
    del ta

    byz = pae.nf.inverse(z_map_train)
    detJ = pae.nf.inverse_log_det_jacobian(z_map_train)
    x = pae.ae.decode(z_map_train)
    reco_error = np.square(x-dataset['x_train'])
    ascore2_train = -0.5*np.dot(reco_error,pae.sigma_square**(-1)) - \
            0.5*np.linalg.norm(byz,axis=1)**2 + detJ
    lp2_train = pae.nf(z_map_train)
    mse2_train = np.mean(reco_error, axis=1)
    mses2_train = np.dot(reco_error,pae.sigma_square**(-1))
tf.profiler.experimental.stop()


#######################################

js_div_lpz2 = mass_sculpting(mjj,-lp2_train)
js_div_pae2 = mass_sculpting(mjj,-ascore2_train)
js_div_mse2 = mass_sculpting(mjj,mse2_train)
js_div_nmse2 = mass_sculpting(mjj,mses2_train)

score = pae.reco_error(dataset['x_train'])
js_div_mse = mass_sculpting(mjj,score)

score = -pae.log_prob_encoding(dataset['x_train'])
js_div_lpz = mass_sculpting(mjj,score)


fig = go.Figure()
# fig.add_shape(
#     type='line', line=dict(dash='dash', color="tomato", width=1),
#     x0=90, x1=90, y0=0, y1=0.04, 
# )

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

fig.write_image(run_dir / "JS.pdf")

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(js_div_pae2.keys()), y=list(js_div_lpz2.values()), mode='lines',
        name=r"$-\log p_{z}^{*}$", line=dict(color="coral", width=3))
)
fig.add_trace(
    go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_mse2.values()), mode='lines',
        name=r"$\text{MSE}$", line=dict(color="royalblue", width=3))
)
fig.add_trace(
    go.Scatter(x=list(js_div_pae.keys()), y=list(js_div_nmse2.values()), mode='lines',
        name=r"$\text{MSE} \cdot \sigma^{\circ-2}$", line=dict(color="deepskyblue", width=3))
)
fig.add_trace(
    go.Scatter(x=list(js_div_pae2.keys()), y=list(js_div_pae2.values()), mode='lines',
        name=r"$\text{PAE}^{*}$", line=dict(color="indigo", width=3))
)
fig.write_image(run_dir / "JS2.pdf")
# In[ ]:


from sklearn.metrics import roc_curve, auc
import plotly.express as px

def make_trace(labels, score, c, n=""):
    fpr, tpr, _ = roc_curve(labels, score)
    aauc = auc(1-fpr, tpr)
    print(n,aauc)
    return go.Scatter(x=tpr, y=1-fpr, mode='lines',
        name=n+f"AUC:{aauc:.2f}", line=dict(color=c, width=2))


def binarize(label):
    return 1 if label == 'sig' else 0
labels = np.array(list(map(binarize, dataset['labels_test'])))

fpr, tpr, _ = roc_curve(labels, ascore)
pae_auc = auc(1-fpr, tpr)

score = pae.reco_error(dataset['x_test'])
roc_mse = make_trace(labels, score, 'steelblue')

score = nmse(dataset['x_test'], pae)
roc_nmse = make_trace(labels, score, 'cornflowerblue')

score = -pae.log_prob_encoding(dataset['x_test'])
roc_lpz = make_trace(labels, score, 'chocolate')

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=tpr, y=1-fpr, mode='lines',
        name=f"AUC:{pae_auc:.2f}", line=dict(color="Plum", width=2))
)

fig.add_trace(roc_mse)
fig.add_trace(roc_nmse)
fig.add_trace(roc_lpz)

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
fig.write_image(run_dir / "roc.pdf")

roc_lpz2 = make_trace(labels, lp2, 'coral')
roc_pae2 = make_trace(labels, ascore2, 'indigo')
roc_mse2 = make_trace(labels, mse2, 'royalblue')
roc_nmse2 = make_trace(labels, mses2, 'deepskyblue')

fig = go.Figure()
fig.add_trace(roc_lpz2)
fig.add_trace(roc_mse2)
fig.add_trace(roc_nmse2)
fig.add_trace(roc_pae2)
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
fig.write_image(run_dir / "roc2.pdf")
