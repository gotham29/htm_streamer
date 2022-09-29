import os
import sys

import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from htm_source.utils import get_args, check_for_missing_features
from htm_source.utils.fs import save_json, load_config, save_config
from htm_source.config import reset_config
from htm_source.pipeline.htm_stream import stream_to_htm


def run_stream(config_path: str,
               data_path: str,
               data_stream_dir: str,
               outputs_dir: str,
               models_dir: str):
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
        os.remove(os.path.join(models_dir, f))

    # 4. For row in Source Data:
    # a. Generate —> ML Inputs (from: row)
    # b. Store —> ML Inputs to ML Inputs Path (from: data_stream_dir)
    # c. Run —> stream_to_htm(ML Input Path, Config Path)
    # d. Delete —> ML Inputs
    try:
        timestep_limit = cfg['timesteps_stop']['running']
    except:
        timestep_limit = 1000000

    print('Running main loop...')
    for idx, row in data[:timestep_limit].iterrows():
        # skip rows where any cfg['features'] aren't numeric
        features_missing = check_for_missing_features(row=row,
                                                      features_expected=cfg['features'])
        if len(features_missing) > 0:
            print(f'\n  skipping row: {idx}')
            print(f'    features_missing = {features_missing}')
            print(f'    data = {dict(row)}\n')
            continue
        # write data
        data_stream_path = os.path.join(data_stream_dir, f'inputrow={idx}.json')
        save_json(dict(row), data_stream_path)
        # call htm module
        stream_to_htm(config_path, data_stream_path, models_dir, outputs_dir)
        # delete data
        os.remove(data_stream_path)
        # print progress
        if idx > 1 and idx % 1000 == 0:
            print(f'  completed row: {idx}')
    # 5. Reset cfg
    cfg = reset_config(cfg)
    cfg = save_config(cfg, config_path)


if __name__ == '__main__':
    args = get_args()
    run_stream(config_path=args.config_path,
               data_path=args.data_path,
               data_stream_dir=args.data_stream_dir,
               models_dir=args.models_dir,
               outputs_dir=args.outputs_dir)
