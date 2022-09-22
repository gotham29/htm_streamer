import os
import sys

import pandas as pd

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_TESTS_DIR, '..')
_DATA_DIR = os.path.join(_REPO_DIR, 'data')
_MODELS_DIR = os.path.join(_TESTS_DIR, 'models')
_OUTPUTS_DIR = os.path.join(_TESTS_DIR, 'results')
_DATA_STREAM_DIR = os.path.join(_TESTS_DIR, 'data')

sys.path.append(_REPO_DIR)

from htm_source.pipeline import run_batch, run_stream
import unittest

unittest.TestLoader.sortTestMethodsUsing = None

os.makedirs(_OUTPUTS_DIR, exist_ok=True)
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_DATA_STREAM_DIR, exist_ok=True)


# print(f'\n_MODELS_DIR = {_MODELS_DIR}')
# print(f'_DATA_STREAM_DIR = {_DATA_STREAM_DIR}')
# print(f'_OUTPUTS_DIR = {_OUTPUTS_DIR}\n')


class IntegrationTests(unittest.TestCase):

    # config_path = os.path.join(_DATA_DIR, 'config.yaml')
    # data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')

    def test_01_run_batch(self):
        config_path = os.path.join(_DATA_DIR, 'config.yaml')
        data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')
        features_models, features_outputs = run_batch(cfg=None,
                                                      config_path=config_path,
                                                      learn=True,
                                                      data=pd.read_csv(data_path),
                                                      iter_print=1000,
                                                      features_models={})
        self.assertIsNone(None)

    def test_02_run_stream(self):
        config_path = os.path.join(_DATA_DIR, 'config.yaml')
        data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')
        result = run_stream(config_path, data_path, _DATA_STREAM_DIR, _OUTPUTS_DIR, _MODELS_DIR)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(failfast=True, exit=True)
