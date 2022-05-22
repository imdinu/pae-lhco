import sys
import glob
from pathlib import Path
from datetime import datetime
sys.path.append("../")

from pae.analysis.scalar_v2 import HLFAnalysis
from pae.models.nn import PaeBuilder

from pae.utils import load_json, dump_json
from pae.loaders.LHCO import ScalarLoaderLHCO, DatasetBuilder

import tensorflow as tf
import numpy as np
# SEED = 42
# np.random.seed(SEED) 
# tf.random.set_seed(SEED)

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

x = ScalarLoaderLHCO.from_json("../pae/configs/loader/bbox1_scalar_2j.json")
mjj = ScalarLoaderLHCO.from_json("../pae/configs/loader/bbox1_scalar_mjj.json")
builder = DatasetBuilder(x, mjj)
builder.data_preparation(sample_sizes ={'box_s':834, 'box_b': 999166, 'bkg': 500_000}, fit_key='bkg')
dataset = builder.make_dataset(train = {'bkg':500_000}, test={'box_s':834, 'box_b': 999166}, replace=True)

files = glob.glob("./test_configs/*")
for i, config_file in enumerate(files):
    print(f"{datetime.now()}[{i+1}/{len(files)}] config_file . . .")
    task = HLFAnalysis(config_file, dataset.copy())
    task.cross_validate()
    del task