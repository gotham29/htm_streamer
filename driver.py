import os

import numpy as np
import pandas as pd
import json

import tqdm

from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_config

CONFIG_PATH = f'NAB/config/config-nab.yaml'
DATA_PATH = os.path.abspath('/home/vlad/Theses/NAB-community/data/')
LABELS_PATH = os.path.abspath('/home/vlad/Theses/NAB-community/labels/combined_windows.json')


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


if __name__ == '__main__':
    # Load config & data
    config = load_config(CONFIG_PATH)
    predictive_features = ['timestamp', 'value']

    with tqdm.tqdm(total=58, desc="Running NAB") as p_bar:
        for root, dirs, files in os.walk(DATA_PATH):
            for f in files:
                if f.endswith('.csv'):
                    if p_bar.n >= 10:
                        exit(0)
                    exp_name = os.path.splitext(f)[0]
                    _, exp_folder = os.path.split(root)
                    result_dir = os.path.join("NAB/results", exp_folder)
                    results_path = os.path.join(result_dir, f"htmStreamer_{f}")
                    os.makedirs(result_dir, exist_ok=True)

                    data = merge_labels_into_data(exp_name)
                    pred_data = data[predictive_features]

                    # Train
                    features_models, features_outputs = run_batch(cfg=config,
                                                                  config_path=None,
                                                                  learn=True,
                                                                  data=pred_data,
                                                                  iter_print=999999,
                                                                  features_models={})

                    # Collect results
                    data['anomaly_score'] = np.array(features_outputs['megamodel_features=2']['anomaly_score'])
                    data.to_csv(results_path)
                    p_bar.update(1)
