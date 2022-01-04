import os
import sys

import numpy as np
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import get_args, save_json
from source.config.config import load_config
from source.pipeline.htm_stream import stream_to_htm


def run_stream(args):

    # 1. Load —> Config from Config Path
    cfg = load_config(args.config_path)
    print(f'\nLoaded —> Config from: {args.config_path}')

    # 2. Load —> ML Inputs from ML Inputs Path
    data = pd.read_csv(args.data_path)
    print(f'Loaded —> Data from: {args.data_path}')

    # 3. For row in Source Data:
        # a. Generate —> ML Inputs (from: row)
        # b. Store —> ML Inputs to ML Inputs Path (from: cfg[‘dir_data’])
        # c. Run —> stream_to_htm(ML Input Path, Config Path)
    print('\nRunning main loop...')
    for _, row in data[:cfg['iter_max']].iterrows():
        # Ensure targets all numeric
        row_numeric = {}
        for k,v in dict(row).items():
            try:
                myfloat = float(v)
                if not np.isnan(myfloat):
                    row_numeric[k] = myfloat
            except:
                pass
        if len(row_numeric) < len(cfg['features']):
            print(f'  skipping row: {_}')
            print(f'    data = {dict(row)}')
            continue
        # write data
        path_data = os.path.join(cfg['dir_data'], f'inputrow={_}.json')
        save_json(dict(row), path_data)
        # call htm module
        stream_to_htm(args.config_path, path_data)
        # print progress
        if _>(cfg['iter_samplesize']*10) and _ % 100 == 0:
            print(f'  completed row: {_}')


if __name__ == '__main__':
    args = get_args()
    run_stream(args)