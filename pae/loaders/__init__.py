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


ALL_FEATURES = ['nj',  'nisj_3', 'nesj_3', 'nc_1', 'nc_2', 'nc_3',
       'tau1_3', 'tau2_3', 'tau3_3', 'tau32_3', 'tau21_3', 'eRing0_3',
       'eRing1_3', 'eRing2_3', 'eRing3_3', 'eRing4_3', 'eRing5_3', 'eRing6_3',
       'eRing7_3', 'eRing8_3', 'eRing9_3', 'nisj_2', 'nesj_2', 'tau1_2',
       'tau2_2', 'tau3_2', 'tau32_2', 'tau21_2', 'eRing0_2', 'eRing1_2',
       'eRing2_2', 'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2',
       'eRing8_2', 'eRing9_2', 'nisj_1', 'nesj_1', 'tau1_1', 'tau2_1',
       'tau3_1', 'tau32_1', 'tau21_1', 'eRing0_1', 'eRing1_1', 'eRing2_1',
       'eRing3_1', 'eRing4_1', 'eRing5_1', 'eRing6_1', 'eRing7_1', 'eRing8_1',
       'eRing9_1', 'pt_3', 'eta_3', 'phi_3', 'mass_3', 'e_3', 'pt_2', 'eta_2',
       'phi_2', 'mass_2', 'e_2', 'pt_1', 'eta_1', 'phi_1', 'mass_1', 'e_1']
       # 'mj1j2', 'mj1j3', 'mj2j3', 'mj1j2j3',

CONTINUOUS_FEATURES = ['tau1_3', 
        'tau2_3', 'tau3_3', 'tau32_3', 'tau21_3', 'eRing0_3', 'eRing1_3',
        'eRing2_3', 'eRing3_3', 'eRing4_3', 'eRing5_3', 'eRing6_3',
        'eRing7_3', 'eRing8_3', 'eRing9_3', 'tau1_2', 'tau2_2', 'tau3_2',
        'tau32_2', 'tau21_2', 'eRing0_2', 'eRing1_2', 'eRing2_2', 'eRing3_2',
        'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2', 'eRing8_2', 
        'eRing9_2', 'tau1_1', 'tau2_1', 'tau3_1', 'tau32_1', 'tau21_1',
        'eRing0_1', 'eRing1_1', 'eRing2_1', 'eRing3_1', 'eRing4_1', 
        'eRing5_1', 'eRing6_1', 'eRing7_1', 'eRing8_1', 'eRing9_1', 'pt_3',
        'eta_3', 'phi_3', 'mass_3', 'e_3', 'pt_2', 'eta_2', 'phi_2',
        'mass_2', 'e_2', 'pt_1', 'eta_1', 'phi_1', 'mass_1', 'e_1']
        #'mj1j2', 'mj1j3', 'mj2j3', 'mj1j2j3', 

DIJET_FEATURES = ['nj', 'nisj_2', 'nesj_2', 'tau1_2', 'tau2_2',  'nc_1',
        'tau3_2', 'tau32_2', 'tau21_2', 'eRing0_2', 'eRing1_2', 'eRing2_2', 
        'eRing3_2', 'eRing4_2', 'eRing5_2', 'eRing6_2', 'eRing7_2', 
        'eRing8_2', 'eRing9_2', 'nisj_1', 'nesj_1', 'tau1_1', 'tau2_1', 
        'tau3_1', 'tau32_1', 'tau21_1', 'eRing0_1', 'eRing1_1', 'eRing2_1', 
        'eRing3_1', 'eRing4_1', 'eRing5_1', 'eRing6_1', 'eRing7_1', 
        'eRing8_1', 'eRing9_1', 'pt_2', 'eta_2', 'phi_2', 'mass_2', 'e_2', 
        'pt_1', 'eta_1', 'phi_1', 'mass_1', 'e_1', 'nc_2']
        # 'mj1j2', 

FEATURE_SETS = {
        'all': ALL_FEATURES,
        'continuous': CONTINUOUS_FEATURES,
        'dijet': DIJET_FEATURES
}

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