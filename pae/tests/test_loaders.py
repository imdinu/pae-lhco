import unittest

import os
import requests
import shutil
from pathlib import Path

from pae.loaders import download_dataset, DATASETS

class TestDatasets(unittest.TestCase):

    def test_links(self):
        for key, url in DATASETS.items():
            try:
                requests.get(url, stream=True, timeout=5)
            except ConnectionError:
                self.fail(f"Dataset {key} url could not be reached")

    def test_download(self):
        path = Path('./data')
        if os.path.exists(path.joinpath('test_tiny')):
            shutil.rmtree(path.joinpath('test_tiny'))
        try:
            download_dataset('test_tiny', path=path)
        except Exception:
            self.fail('Exception raised during dataset download')

        path = path.joinpath('test_tiny')
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.isfile(path.joinpath('test_tiny.tar')))
        self.assertTrue(os.path.isfile(path.joinpath('bkgHLF_merged.h5')))
        self.assertTrue(os.path.isfile(path.joinpath('sigHLF_merged.h5')))

if __name__ == '__main__':
    unittest.main()