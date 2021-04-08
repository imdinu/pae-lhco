"""Module of data loading and dataset creation tools.

This module contains data loaders for LHCO datasets. There are also constant 
lists defined for various feature sets and scalers, which are merged toghether 
dictionaries."""
import os

import requests
import tarfile
from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, \
                                  StandardScaler, RobustScaler, Normalizer, \
                                  MaxAbsScaler    

CONTINUOUS_FEATURES = ['pt1', 'eta1', 'phi1', 'E1', 'm1', '1tau1', '2tau1', 
        '3tau1', '32tau1', '21tau1', 'pt2', 'eta2', 'phi2', 'E2', 'm2',
        '1tau2', '2tau2', '3tau2', '32tau2', '21tau2',
        'eRing0_1', 'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 'eRing5_1',
        'eRing6_1', 'eRing7_1', 'eRing8_1', 'eRing9_1', 'eRing0_2', 'eRing1_2',
        'eRing2_2', 'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2',
        'eRing8_2', 'eRing9_2', 'mjj']

ALL_FEATURES = ['pt1', 'eta1', 'phi1', 'E1', 'm1', 'nc1', 'nisj1', 'nesj1', 
        '1tau1', '2tau1', '3tau1', '32tau1', '21tau1', 'pt2', 'eta2', 'phi2',
        'E2', 'm2', 'nc2', 'nisj2', 'nesj2', '1tau2', '2tau2', '3tau2', 
        '32tau2', '21tau2',
        'eRing0_1', 'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 'eRing5_1',
        'eRing6_1', 'eRing7_1', 'eRing8_1', 'eRing9_1', 'eRing0_2', 'eRing1_2',
        'eRing2_2', 'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2',
        'eRing8_2', 'eRing9_2', 'nj']

TRIMMED_FEATURES = ['1tau1', '1tau2', '21tau1', '21tau2', '32tau1', '32tau2',
        'eRing0_1', 'eRing0_2', 'eRing1_1', 'eRing1_2', 'eRing3_1', 'eRing3_2',
        'm1', 'm2', 'nc1', 'nc2', 'nisj1', 'nisj2', 'pt1', 'pt2', 'mjj']

MINIMAL_FEATURES = ['pt1', 'eta1', 'E1', 'm1', '21tau1', '32tau1', 
        'pt2', 'eta2', 'E2', 'm2', '21tau2', '32tau2', 'mjj']

TRIJET_ALL = ['pt1', 'eta1', 'phi1', 'E1', 'm1', 'nc1', 'nisj1', 'nesj1', 
       '1tau1', '2tau1', '3tau1', '32tau1', '21tau1', 'pt2', 'eta2', 'phi2', 
       'E2', 'm2', 'nc2', 'nisj2', 'nesj2', '1tau2', '2tau2', '3tau2', 
       '32tau2', '21tau2', 'pt3', 'eta3', 'phi3', 'E3', 'm3', 'nc3', 'nisj3', 
       'nesj3', '1tau3', '2tau3', '3tau3', '32tau3', '21tau3', 'eRing0_1', 
       'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 'eRing5_1', 'eRing6_1', 
       'eRing7_1', 'eRing8_1', 'eRing9_1', 'eRing0_2', 'eRing1_2', 'eRing2_2', 
       'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2', 'eRing8_2', 
       'eRing9_2', 'eRing0_3', 'eRing1_3', 'eRing2_3', 'eRing3_3', 'eRing4_3', 
       'eRing5_3', 'eRing6_3', 'eRing7_3', 'eRing8_3', 'eRing9_3', 'nj', 'mjj', 
       'mjjj']

FEATURE_SETS = {
        'all': ALL_FEATURES,
        'continuous': CONTINUOUS_FEATURES,
        'trimmed': TRIMMED_FEATURES,
        'minimal': MINIMAL_FEATURES,
        'trijet': TRIJET_ALL}

SCALERS = {
        'min_max': MinMaxScaler,
        'standard': StandardScaler,
        'quantile': QuantileTransformer,
        'robust': RobustScaler,
        'normalizer': Normalizer,
        'max_abs': MaxAbsScaler}

DATASETS = {
        'RnD': 'https://cernbox.cern.ch/index.php/s/qTPpq0uHvwYKWqM/download',
        'test_tiny': 'https://cernbox.cern.ch/index.php/s/v0dIs1wcKYAPXSl/download',
}

DATASETS_IMG = {
        'RnD_images_sig': 'https://cernbox.cern.ch/index.php/s/lBIu16xOI4P9hf9/download',
        'Rnd_images_bkg': 'https://cernbox.cern.ch/index.php/s/5TlR3C4htOEj8FT/download',
}

def download_dataset(key, path="../data/", images=False):
    """Downlads the requested dataset from cerbox based on key. 

    All of the available keys can be found in the `DATASETS` dictionary from
    this module

    Args:
        key (str): String literal of a valid dataset identifier.
        path (Path): Path to folder where the data will be downloaded.
            A secondary folder will be created in this directory named after 
            the daataset key.
        images (bool): Wheter or not to also download jet image data
    Returns:
        None
    """

    # Retrive the dataset link
    try:
        dataset_url = DATASETS[key]
    except KeyError:
        print(f"'{key}' is not a valid dataset, available datasets for "
              f"download are: {list(DATASETS.keys())}")
        raise

    # Create folder structure (if necessary)
    store_path = Path(path).joinpath(key.split('_')[0])
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    file_path = store_path.joinpath(f'{key}.tar')

    # Download the file
    descriptor = f"Downloading {key} data"
    download_file(dataset_url, file_path, descriptor)

    #Untar files
    tar_file = tarfile.open(file_path, "r:*")
    tar_file.extractall(file_path.parent)
    tar_file.close()

    # Cleanup
    os.remove(file_path)

    # Download images
    if images:
        for img_label in DATASETS_IMG.keys():
            if key in img_label:
                descriptor = f"Downloading {img_label} data"
                file_path = store_path.joinpath(f'{img_label}.h5')
                download_file(dataset_url, file_path, descriptor=descriptor, 
                              timeout=None)

def download_file(url, path, descriptor=None, chunk_size=1048576, timeout=None):
    """Downloads a file to the secified path
    
    Args:
        url (str): URL of the file
        path (Path): The location where the file will be saved
        descriptor (string): Progres bar annotation
        chunksize  (int): Number of bytes per chunk
        timeout (float or tuple): Seconds to wait for the response

    Returns:
        None
    """
    ans = requests.get(url, stream=True, timeout=timeout)
    with open(path, "wb") as file:
        for chunk in tqdm(ans.iter_content(chunk_size=1024**2),
                          unit='kB',
                          desc=descriptor):
            if chunk:
                file.write(chunk)