import argparse
import json
import numpy as np
import os
import pandas as pd
import tqdm
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_config


def print_config(config, indents=2):
    indent = ' ' * indents
    for k, v in config.items():
        print(f"\n{indent}{k}")
        for k_, v_ in v.items():
            print(f"{indent}  {k_}")
            if isinstance(v_, dict):
                for k__, v__ in v_.items():
                    print(f"{indent}    {k__} = {v__}")
            else:
                print(f"{indent}    = {v_}")


def get_labels(ds_name):
    with open(LABELS_PATH, 'r') as f:
        tags = json.loads(f.read())

    try:
        tag_file = list(filter(lambda x: ds_name in x, tags.keys()))[0]
    except IndexError:
        raise NameError(f"No tags found for {ds_name}")
    else:
        anom_positions = tags[tag_file]

    return anom_positions


def get_data(ds_name):
    ds_file = None
    for root, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if ds_name in f:
                if ds_file:
                    raise NameError(f"{ds_name} is ambigious")

                ds_file = os.path.join(root, f)

    if not ds_file:
        raise NameError(f"{ds_name} not found")

    return pd.read_csv(ds_file)


def merge_labels_into_data(ds_name):
    df = get_data(ds_name)
    tags = get_labels(ds_name)

    df['label'] = 0
    df['anomaly_score'] = None
    for start, end in tags:
        criteria = f'"{start}" <= timestamp <= "{end}"'
        df.loc[df.eval(criteria), 'label'] = 1

    return df


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d_htm', '--dir_htm', required=True,
                        help='path to htm_streamer dir')

    parser.add_argument('-d_nab', '--dir_nab', required=True,
                        help='path to NAB dir')

    return parser.parse_args()


if __name__ == '__main__':

    # Get args
    args = get_args()
    DATASET_COUNT = 60
    PROBATION_PERCENT = 0.1
    dir_htm = args.dir_htm  # "/Users/samheiserman/Desktop/repos/htm_streamer"
    dir_nab = args.dir_nab  # "/Users/samheiserman/Desktop/repos/NAB"

    CONFIG_PATH = os.path.join(dir_htm, 'data', 'config--nab.yaml')
    CONFIG_DEF_PATH = os.path.join(dir_htm, 'data', 'config--default.yaml')
    RESULTS_DIR = os.path.join(dir_nab, 'results', 'htmStreamer')
    DATA_PATH = os.path.join(dir_nab, 'data')
    LABELS_PATH = os.path.join(dir_nab, 'labels', 'combined_windows.json')

    # Load config & data
    config = load_config(CONFIG_PATH)
    config_def = load_config(CONFIG_DEF_PATH)
    print_config(config)
    print_config(config_def)

    predictive_features = ['timestamp', 'value']

    with tqdm.tqdm(total=58, desc="\n\nRunning NAB") as p_bar:
        for root, dirs, files in os.walk(DATA_PATH):
            for f in files:
                if f.endswith('.csv'):
                    if p_bar.n >= DATASET_COUNT:
                        exit(0)
                    exp_name = os.path.splitext(f)[0]
                    _, exp_folder = os.path.split(root)
                    exp_folder_dir = os.path.join(RESULTS_DIR, exp_folder)
                    os.makedirs(os.path.join(exp_folder_dir), exist_ok=True)
                    results_path = os.path.join(exp_folder_dir, f"htmStreamer_{f}")
                    data = merge_labels_into_data(exp_name)
                    pred_data = data[predictive_features]
                    # Train
                    print(f"\n\n{f}")
                    config_def['models_params']['alikl']['probationaryPeriod'] = int(data.shape[0]*PROBATION_PERCENT)
                    features_models, features_outputs = run_batch(cfg=config,
                                                                  cfg_default=config_def,
                                                                  config_path=None,
                                                                  config_default_path=None,
                                                                  learn=True,
                                                                  data=pred_data,
                                                                  iter_print=999999,
                                                                  features_models={})
                    # Collect results
                    data['anomaly_score'] = np.array(
                        features_outputs[f'megamodel_features={len(predictive_features)}']['anomaly_likelihood'])
                    data.to_csv(results_path)
                    p_bar.update(1)
