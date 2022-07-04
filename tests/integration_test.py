import os
import pandas as pd
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_TESTS_DIR, '..')
_DATA_DIR = os.path.join(_REPO_DIR, 'data')
_MODELS_DIR = os.path.join(_TESTS_DIR, 'models')
_OUTPUTS_DIR = os.path.join(_TESTS_DIR, 'results')
_DATA_STREAM_DIR = os.path.join(_TESTS_DIR, 'data')

sys.path.append(_REPO_DIR)

from htm_source.pipeline.htm_stream_runner import run_stream
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.utils import make_dir
import unittest

unittest.TestLoader.sortTestMethodsUsing = None

make_dir(_OUTPUTS_DIR)
make_dir(_MODELS_DIR)
make_dir(_DATA_STREAM_DIR)

# print(f'\n_MODELS_DIR = {_MODELS_DIR}')
# print(f'_DATA_STREAM_DIR = {_DATA_STREAM_DIR}')
# print(f'_OUTPUTS_DIR = {_OUTPUTS_DIR}\n')


class IntegrationTests(unittest.TestCase):

    # config_path = os.path.join(_DATA_DIR, 'config.yaml')
    # data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')

    def test_01_run_batch(self):
        config_path = os.path.join(_DATA_DIR, 'config.yaml')
        data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')
        subjects_models[subj], subj_outputs = run_batch(cfg=None,
                                                        config_path=config_path,
                                                        learn=True,
                                                        data=pd.read_csv(data_path),
                                                        iter_print=1000,
                                                        features_models={})

    def test_02_run_stream(self):
        config_path = os.path.join(_DATA_DIR, 'config.yaml')
        data_path = os.path.join(_DATA_DIR, 'batch', 'sample_timeseries.csv')
        result = run_stream(config_path, data_path, _DATA_STREAM_DIR, _OUTPUTS_DIR, _MODELS_DIR)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(failfast=True, exit=True)
