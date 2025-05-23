import argparse
import json
import numpy as np
import os
import pandas as pd
import tqdm
from htm_streamer.pipeline.htm_batch_runner import run_batch
from htm_streamer.utils.fs import load_config

from logger import setup_applevel_logger
log = setup_applevel_logger(file_name= os.path.join(os.getcwd(),'logs.log') )


def print_config(config, indents=2):
    log.info(msg="  config:")
    indent = ' ' * indents
    for k, v in config.items():
        log.info(msg=f"    {indent}{k}")
        for k_, v_ in v.items():
            log.info(msg=f"      {indent}  {k_}")
            if isinstance(v_, dict):
                for k__, v__ in v_.items():
                    log.info(msg=f"      {indent}    {k__} = {v__}")
            else:
                log.info(msg=f"      {indent}    = {v_}")


def get_labels(ds_name, path_labels):
    with open(path_labels, 'r') as f:
        tags = json.loads(f.read())

    try:
        tag_file = list(filter(lambda x: ds_name in x, tags.keys()))[0]
    except IndexError:
        raise NameError(f"No tags found for {ds_name}")
    else:
        anom_positions = tags[tag_file]

    return anom_positions


def get_data(ds_name, path_data):
    ds_file = None
    for root, dirs, files in os.walk(path_data):
        for f in files:
            if ds_name in f:
                if ds_file:
                    raise NameError(f"{ds_name} is ambigious")

                ds_file = os.path.join(root, f)

    if not ds_file:
        raise NameError(f"{ds_name} not found")

    return pd.read_csv(ds_file)


def merge_labels_into_data(ds_name, path_data, path_labels):
    df = get_data(ds_name, path_data)
    tags = get_labels(ds_name, path_labels)

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


def run_nab(dir_htm, dir_nab, dataset_count, probation_proportion):
    log.info("\n\n**run_nab()")
    config_path_user = os.path.join(dir_htm, 'config', 'config--nab.yaml')
    config_path_model = os.path.join(dir_htm, 'config', 'config--model_default.yaml')
    dir_results = os.path.join(dir_nab, 'results', 'htmStreamer')
    path_data = os.path.join(dir_nab, 'data')
    path_labels = os.path.join(dir_nab, 'labels', 'combined_windows.json')

    # Load config & data
    cfg_user = load_config(config_path_user)
    cfg_model = load_config(config_path_model)
    print_config(cfg_user)
    print_config(cfg_model)

    predictive_features = ['timestamp', 'value']

    with tqdm.tqdm(total=58, desc="\n\nRunning NAB") as p_bar:
        for root, dirs, files in os.walk(path_data):
            files_csv = [f for f in files if f.endswith('.csv')]
            for f in files_csv:
                # if f.endswith('.csv'):
                if p_bar.n >= dataset_count:
                    exit(0)
                exp_name = os.path.splitext(f)[0]
                _, exp_folder = os.path.split(root)
                exp_folder_dir = os.path.join(dir_results, exp_folder)
                os.makedirs(os.path.join(exp_folder_dir), exist_ok=True)
                results_path = os.path.join(exp_folder_dir, f"htmStreamer_{f}")
                data = merge_labels_into_data(exp_name, path_data, path_labels)
                pred_data = data[predictive_features]
                # Train
                log.info(f"\n  file = {f}")
                cfg_model['models_params']['anomaly_likelihood']['probationaryPeriod'] = int(
                    data.shape[0] * probation_proportion)
                features_models, features_outputs = run_batch(cfg_user=cfg_user,
                                                              cfg_model=cfg_model,
                                                              config_path_user=None,
                                                              config_path_model=None,
                                                              learn=True,
                                                              data=pred_data,
                                                              iter_print=999999,
                                                              features_models={})
                # Collect results
                data['anomaly_score'] = np.array(
                    features_outputs[f'megamodel_features={len(predictive_features)}']['anomaly_likelihood'])
                data['raw_score'] = np.array(
                    features_outputs[f'megamodel_features={len(predictive_features)}']['anomaly_score'])
                data.to_csv(results_path)
                p_bar.update(1)


if __name__ == '__main__':
    # Get args
    args = get_args()
    run_nab(args.dir_htm, args.dir_nab, dataset_count=60, probation_proportion=0.1)
