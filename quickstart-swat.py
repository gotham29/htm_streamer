import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from htm_source.model.htm_pyramid import ModelPyramid
from htm_source.utils.fs import load_config
from htm_source.utils.metric import find_best_fb, merge_targets_and_predictions, plot_windows
import logging

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Load config & data
    config_path_user = os.path.join(os.getcwd(), 'config', 'config--swat.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = "/home/vlad/Theses/SCADA-data/SWaT.A1_A2_Dec_2015/Physical/SWaT_Dataset_Full_v0_Cleaned.parquet"
    data_cfg = load_config(config_path_user)
    model_cfg = load_config(config_path_model)
    full_data = pd.read_parquet(data_path)

    res_data = 10
    min_data = 300_000
    max_data = 600_000
    learn_period = 15_000
    redundant_col = ['p202', 'p401', 'p404', 'p502', 'p601', 'p603']

    merge_mode = 'sd'
    layer_inputs = 3
    max_pool = 2

    data = full_data.copy().iloc[min_data:max_data:res_data].reset_index(drop=True)
    X = data[[c for c in data.columns if c not in ['label', 'timestamp'] + redundant_col]]

    model = ModelPyramid(data=X,
                         inputs_per_layer=layer_inputs,
                         feat_cfg=data_cfg['features'],
                         enc_cfg=model_cfg['models_encoders'],
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'],
                         feature_mode='split',
                         # num_choices=81,
                         anomaly_score=True,
                         al_learn_period=learn_period,
                         max_pool=max_pool,
                         flatten=True,
                         merge_mode=merge_mode)

    model.model_summary()
    model.run()

    pred = np.array(model.head.anomaly['likelihood'])

    # pred = pd.read_csv('eye-state-results-nab.csv')['anomaly_score'].values

    gt = data['label'].values

    thresh = np.arange(0.7, 0.95, 0.01)
    winds = np.arange(5, 55, 5)

    score, (t, w) = find_best_fb(pred, gt, thresh, winds, learn_period=learn_period, beta=0.5, pred_window_size=2)
    if score > 0:
        pred_merged, target_merged = merge_targets_and_predictions(pred, gt, t, w, learn_period)
        plot_windows(pred_merged, target_merged, pred, gt, t, learn_period)
    else:
        print("The model didn't learn, best score is 0!")

    # model.plot_results(data['eyeDetection'])

    print(1)
