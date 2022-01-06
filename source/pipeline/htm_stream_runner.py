import os
import sys

import numpy as np
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import get_args, save_json
from source.config.config import load_config, save_config
from source.pipeline.htm_stream import stream_to_htm


def run_stream(args):

    # 1. Load —> Config from Config Path
    cfg = load_config(args.config_path)
    print(f'\nLoaded —> Config from: {args.config_path}')

    # 2. Load —> ML Inputs from ML Inputs Path
    data = pd.read_csv(args.data_path)
    print(f'Loaded —> Data from: {args.data_path}')

    # 3. Delete any .pkl in dir_models
    pkl_files = [f for f in os.listdir(cfg['dir_models']) if '.pkl' in f]
    for f in pkl_files:
        os.remove( os.path.join(cfg['dir_models'], f))

    # 4. For row in Source Data:
        # a. Generate —> ML Inputs (from: row)
        # b. Store —> ML Inputs to ML Inputs Path (from: cfg[‘dir_data’])
        # c. Run —> stream_to_htm(ML Input Path, Config Path)
    print('\nRunning main loop...')
    for _, row in data[:cfg['iter_stop']].iterrows():
        # check which features are numeric
        features_numeric = {}
        for f, v in dict(row).items():
            try:
                f_float = float(v)
                if not np.isnan(f_float):
                    features_numeric[f] = f_float
            except:
                pass
        # skip rows where any cfg['features'] aren't numeric
        features_missing = [f for f in cfg['features'] if f not in features_numeric]
        if len(features_missing) > 0:
            print(f'  skipping row: {_}')
            print(f'    data = {dict(row)}')
            continue
        # write data
        path_data = os.path.join(cfg['dir_data'], f'inputrow={_}.json')
        save_json(dict(row), path_data)
        # call htm module
        stream_to_htm(args.config_path, path_data)
        # print progress
        if _ > (cfg['iter_samplesize']*10) and _ % 100 == 0:
            print(f'  completed row: {_}')

    # 5. Delete stream data files
    print('\nRemoving stream files...')
    json_files = [f for f in os.listdir(cfg['dir_data'])]
    for f in json_files:
        os.remove( os.path.join(cfg['dir_data'], f))

    # 6. Reset Config:
    #     a. iter_current —> 0
    #     b. learn —> True
    #     c. mode --> sample_data
    #     d. features_samples --> deleted
    print('\nResetting config...')
    cfg['iter_current'] = 0
    cfg['learn'] = True
    cfg['mode'] = 'sample_data'
    # del cfg['features_samples']
    cfg = {k:v for k,v in cfg.items() if k != 'features_samples'}
    print('  iter_current = 0')
    print('  learn = True')
    print('  mode = sample_data')
    print('  features_samples --> deleted')
    save_config(cfg, args.config_path)


if __name__ == '__main__':
    args = get_args()
    run_stream(args)