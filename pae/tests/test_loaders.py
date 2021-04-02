import unittest
import requests
from pae.loaders import download_dataset, DATASETS

class TestLoaders(unittest.TestCase):

    def test_links(self):
        for url in DATASETS.values():
            self.assertRaises(ConnectionError, requests.get, url)

if __name__ == '__main__':
    unittest.main()