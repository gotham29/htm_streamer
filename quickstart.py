import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from htm_source.pipeline.htm_batch_runner import run_batch, new_run
from htm_source.utils.fs import load_config

if __name__ == '__main__':
    # Specify run params
    # model_for_each_feature = False
    # return_pred_count = False
    # timestep_tostop_sampling = 40
    # timestep_tostop_learning = 4000
    # timestep_tostop_running = 5000

    # Load config & data
    config_path_user = os.path.join(os.getcwd(), 'config', 'config--datagen.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = "/home/vlad/Theses/datagen/data_fs-96_cyc-100.csv"
    cfg_user = load_config(config_path_user)
    cfg_model = load_config(config_path_model)
    data = pd.read_csv(data_path)
    X = data[['x', 'y']].iloc[:]
    y = data['x']
    X['x'].loc[6000:6015] *= np.random.rand(16)
    # cfg_user['timesteps_stop']['sampling'] = timestep_tostop_sampling
    # cfg_user['timesteps_stop']['learning'] = timestep_tostop_learning
    # cfg_user['timesteps_stop']['running'] = timestep_tostop_running
    # cfg_user['models_state']['model_for_each_feature'] = model_for_each_feature
    # cfg_user['models_state']['return_pred_count'] = return_pred_count

    # Train
    # features_models, features_outputs = run_batch(cfg_user=cfg_user,
    #                                               cfg_model=cfg_model,
    #                                               config_path_user=None,
    #                                               config_path_model=None,
    #                                               learn=True,
    #                                               data=X,
    #                                               iter_print=100,
    #                                               features_models={})

    model_x, model_y, model_xy = new_run(data=X,
                                         data_cfg=cfg_user,
                                         model_cfg=cfg_model)

    n = 15000
    plt.plot(model_x.anomaly['score'][:n])
    plt.plot(model_y.anomaly['score'][:n])
    # plt.plot(features_outputs['y']['anomaly_likelihood'][:n])
    plt.show()

    plt.plot(model_xy.anomaly['score'])
    plt.show()
    print(1)
    # Collect results
    # f1 = features[0]
    # f1_model = features_models[f1]
    # f1_anomaly_scores = features_outputs[f1]['anomaly_score']
    # f1_anomaly_liklihoods = features_outputs[f1]['anomaly_likelihood']
    # f1_prediction_counts = features_outputs[f1]['pred_count']
