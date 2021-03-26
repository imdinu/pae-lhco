import os
import sys
import glob
import argparse

import tensorflow as tf
import numpy as np

sys.path.append("../")

from loaders.LHCO import LhcoRnDLoader
from analysis.scalar import HLFAnalysis
from models.nn import PaeBuilder
from utils import load_json, dump_json

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs_dir", action='store', 
                        type=str, )
    parser.add_argument("--out_dir", action='store', 
                        type=str, default='./scan_results/')

    args = parser.parse_args()
    config_files = glob.glob(args.configs_dir+'/*.json')
    config_files

    plots_dir = args.out_dir
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    results_history = {
        'id': [],
        'config': [],
        'js_div_train': [],
        'js_div_test': [],
        'sig_eff': [],
        'bkg_rej': [],
        'auc': []
    }

    for file in config_files:
        analysis_cfg = load_json(file)
        analysis_cfg

        config = analysis_cfg.copy()

        loader_json = analysis_cfg.pop('ANA:loader')
        dataset_json = analysis_cfg.pop('ANA:dataset')
        density_estimator = analysis_cfg.pop('ANA:estimator')
        prc = analysis_cfg.pop('ANA:percentile')

        loader = LhcoRnDLoader.from_json(loader_json)
        dataset_cfg = load_json(dataset_json)
        dataset = loader.make_full_dataset(**dataset_cfg)

        builder = PaeBuilder()

        pae, ae_train, nf_train = builder.from_json(analysis_cfg)
        task = HLFAnalysis(pae, dataset=dataset)
        task.reweighting(estimator=density_estimator, fit_key='mjj_train')
        if 'cond' in config['nf_model']:
            task.make_cond_inputs(['mjj_train', 'mjj_test', 'mjj_valid'])
        task.train(ae_train,nf_train)
        result = task.evaluate(prc = prc)
        for key in result.keys():
            results_history[key].append(result[key])
        
        id = config_files[0].replace('.', '/').split('/')[-2]
        results_history['id'].append(id)
        results_history['config'].append(config)
        task.plot_training(plots_dir+id+'_train.png')
        task.plot_latent_space(plots_dir+id+'_latent.png')

    dump_json(results_history, plots_dir+'restult.json')


