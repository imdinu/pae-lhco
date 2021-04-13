import unittest

import os
import requests
import shutil
from pathlib import Path

import numpy as np

from pae.loaders import download_dataset, DATASETS
from pae.loaders.LHCO import ScalarLoaderLHCO, ImageLoaderLHCO, DatasetBuilder

class TestDatasets(unittest.TestCase):

    def test_links(self):
        for key, url in DATASETS.items():
            try:
                requests.get(url, stream=True, timeout=5)
            except ConnectionError:
                self.fail(f"Dataset {key} url could not be reached")

    def test_download(self):
        path = Path('./data')
        if os.path.exists(path.joinpath('test')):
            shutil.rmtree(path.joinpath('test'))
        try:
            download_dataset('test_tiny', path=path)
        except Exception:
            self.fail('Exception raised during dataset download')

        path = path.joinpath('test')
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.isfile(path.joinpath('test_tiny.tar')))
        self.assertTrue(os.path.isfile(path.joinpath('RnD_scalars_bkg.h5')))
        self.assertTrue(os.path.isfile(path.joinpath('RnD_scalars_sig.h5')))

class TestLoaders(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        file_paths = {
            "bkg": "data/test/RnD_scalars_bkg.h5",
            "sig": "data/test/RnD_scalars_sig.h5"
        }
        file_paths_img = {
            "bkg": "data/RnD/bkg_imgs.h5",
            "sig": "data/RnD/sig_imgs.h5"
        }

        cls.loader_all = ScalarLoaderLHCO(file_paths, features='all', 
                                           scaler='min_max', name='x')
        cls.loader_mjj = ScalarLoaderLHCO(file_paths, features=['mjj'], 
                                           scaler=None, name='mjj')
        cls.loader_cfg = ScalarLoaderLHCO.from_json(
                                "pae/configs/loader/scalar_test.json")

        cls.loader_img = ImageLoaderLHCO(file_paths_img, name='x')
        cls.sample_size = {'sig':500, 'bkg': 1000}
        cls.builder = DatasetBuilder()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.loader_all.load_events(self.sample_size)
        self.loader_mjj.load_events()
        self.loader_cfg.load_events(self.sample_size)
        self.loader_img.load_events(self.sample_size)

    def tearDown(self):
        pass

    def test_load(self):
        self.assertEqual(len(self.loader_mjj), 10_000)
        self.assertEqual(len(self.loader_all), 1_500)
        self.assertEqual(len(self.loader_cfg), 1_500)
        self.assertEqual(len(self.loader_img), 1_500)

        bkg = sum(self.loader_mjj._available_events['bkg'])
        sig = sum(self.loader_mjj._available_events['sig'])
        self.assertEqual(bkg, 9128)
        self.assertEqual(sig, 872)

    def test_rescale(self):
        self.loader_all.rescale('bkg')
        self.assertAlmostEqual(self.loader_all['bkg'].max(), 1, 10)
        
        self.assertRaises(AttributeError, self.loader_mjj.rescale, 'bkg')
        self.assertRaises(NotImplementedError, self.loader_img.rescale)

        self.loader_cfg.rescale('bkg')
        bkg = self.loader_cfg['bkg']
        sig = self.loader_cfg['sig']
        self.assertAlmostEqual(bkg.max(), 1, 10)
        self.assertEqual(bkg.mean(), 0.5)
        self.assertGreater(bkg.std(), 0.25)
        self.assertGreater(sig.std(), 0.25)

    def test_make_dataset(self):
        indices_cfg = {'bkg':np.arange(128), 'sig':np.arange(72)}
        data_cfg = self.loader_cfg[indices_cfg]
        self.assertEqual(len(data_cfg), 200)

        data_img = self.loader_img[indices_cfg]
        self.assertEqual(data_img.shape, (200, 64, 64, 1))

        indices_all = {'train':{'bkg': np.arange(500,1000)}, 
                       'test':{'bkg': np.arange(500), 'sig':np.arange(500)}}

        dataset_all = self.loader_all.make_dataset(indices_all)
        self.assertListEqual(list(dataset_all.keys()), ['x_train', 'x_test'])
        self.assertEqual(len(dataset_all['x_train']), 500)
        self.assertEqual(len(dataset_all['x_test']), 1000)

        dataset_mjj = self.loader_mjj.make_dataset(indices_all)
        self.assertListEqual(list(dataset_mjj.keys()), 
                             ['mjj_train', 'mjj_test'])

    def test_dataset_builder(self):
        builder = DatasetBuilder(self.loader_img, self.loader_mjj)
        builder.data_preparation(sample_sizes=self.sample_size)

        d0 = builder.make_dataset(train={'bkg': 1_000}, validation_split=0.33333)
        self.assertEqual(d0['x_train'].shape, (667, 64, 64, 1))
        self.assertRaises(RuntimeError, builder.make_dataset, None, {'bkg': 1_000}, 0)
        self.assertLessEqual(list(d0.keys()), ['x_train', 'mjj_train', 'labels_train'])

        dr = builder.make_dataset(test={'sig': 1_000}, validation_split=0, replace=True)
        self.assertLessEqual(len(np.unique(dr['x_test'], axis=0)), 500)

if __name__ == '__main__':
    unittest.main()