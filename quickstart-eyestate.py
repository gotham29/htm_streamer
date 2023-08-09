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
    config_path_user = os.path.join(os.getcwd(), 'config', 'config--eyestate.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = "/home/vlad/Theses/eye-state/eye-state.csv"
    data_cfg = load_config(config_path_user)
    model_cfg = load_config(config_path_model)

    learn_period = 1000
    merge_mode = 'sd'
    data = pd.read_csv(data_path).loc[:]
    X = data[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']]

    model = ModelPyramid(data=X,
                         inputs_per_layer=3,
                         feat_cfg=data_cfg['features'],
                         enc_cfg=model_cfg['models_encoders'],
                         sp_cfg=model_cfg['models_params']['sp'],
                         tm_cfg=model_cfg['models_params']['tm'],
                         seed=model_cfg['models_params']['seed'],
                         anomaly_score=True,
                         al_learn_period=learn_period,
                         max_pool=2,
                         flatten=True,
                         merge_mode=merge_mode)

    model.run()

    pred = np.array(model.head.anomaly['likelihood'])

    # pred = pd.read_csv('eye-state-results-nab.csv')['anomaly_score'].values

    gt = data['eyeDetection'].values

    thresh = np.arange(0.5, 1, 0.01)
    winds = np.arange(5, 55, 5)

    score, (t, w) = find_best_fb(pred, gt, thresh, winds, learn_period=learn_period, beta=1)
    pred_merged, target_merged = merge_targets_and_predictions(pred, gt, t, w, learn_period)
    plot_windows(pred_merged, target_merged, pred, gt, t, learn_period)
    # model.plot_results(data['eyeDetection'])

    print(1)
