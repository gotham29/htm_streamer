import os
import sys

import numpy as np
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import get_args, save_json
from source.config.config import load_config, save_config
from source.pipeline.htm_stream import stream_to_htm


def run_stream(config_path, data_path, data_stream_dir, outputs_dir, models_dir):
    """
    Purpose:
        Loop over all rows in batch csv
            generate inputs for 'stream_to_htm'
            run 'stream_to_htm'
    Inputs:
        config_path
            type: str
            meaning: path to config (yaml)
        data_path
            type: str
            meaning: path to batch data (csv)
        data_stream_dir
            type: str
            meaning: path to dir where stream datas (json) are written to
        outputs_dir
            type: str
            meaning: path to dir where 'stream_to_htm' outputs are written to
        models_dir
            type: str
            meaning: path to dir where 'stream_to_htm' models are written to
    Outputs:
        stream datas --> written to 'data_stream_dir'
        'stream_to_htm' --> called for each stream data
        'config.yaml' --> reset after last stream data
    """

    # 1. Load —> Config from Config Path
    cfg = load_config(config_path)
    print(f'\nLoaded —> Config from: {config_path}')

    # 2. Load —> ML Inputs from ML Inputs Path
    data = pd.read_csv(data_path)
    print(f'Loaded —> Data from: {data_path}')

    # 3. Delete any .pkl in dir_models
    pkl_files = [f for f in os.listdir(models_dir) if '.pkl' in f]
    for f in pkl_files:
        os.remove( os.path.join(models_dir, f))

    # 4. For row in Source Data:
        # a. Generate —> ML Inputs (from: row)
        # b. Store —> ML Inputs to ML Inputs Path (from: data_stream_dir)
        # c. Run —> stream_to_htm(ML Input Path, Config Path)
    print('\nRunning main loop...')
    for _, row in data[:cfg['timesteps_stop']['running']].iterrows():
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
        data_stream_path = os.path.join(data_stream_dir, f'inputrow={_}.json')
        save_json(dict(row), data_stream_path)
        # call htm module
        stream_to_htm(config_path, data_stream_path, models_dir, outputs_dir)
        # print progress
        if _ > (cfg['timesteps_stop']['sampling']*10) and _ % 100 == 0:
            print(f'  completed row: {_}')

    # 5. Delete stream data files
    print('\nRemoving stream files...')
    json_files = [f for f in os.listdir(data_stream_dir)]
    for f in json_files:
        os.remove( os.path.join(data_stream_dir, f))

    # 6. Reset Config:
    #     a. models_state['timestep'] —> 0
    #     b. models_state['learn'] —> True
    #     c. models_state['mode'] --> sample_data
    #     d. features_samples --> deleted
    print('\nResetting config...')
    cfg['models_state']['timestep'] = 0
    cfg['models_state']['learn'] = True
    cfg['models_state']['mode'] = 'sample_data'
    # del cfg['features_samples']
    cfg = {k:v for k,v in cfg.items() if k != 'features_samples'}
    print('  timestep = 0')
    print('  learn = True')
    print('  mode = sample_data')
    print('  features_samples --> deleted')
    save_config(cfg, config_path)


if __name__ == '__main__':
    args = get_args()
    run_stream(args.config_path, args.data_path, args.models_dir, args.outputs_dir)
