import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from htm_source.pipeline.htm_batch_runner import corr_run2
from htm_source.utils.fs import load_config

if __name__ == '__main__':
    # Load config & data
    config_path_user = os.path.join(os.getcwd(), 'config', 'config--datagen-corr2.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = "/home/vlad/Theses/datagen/corr_seq2"
    cfg_user = load_config(config_path_user)
    cfg_model = load_config(config_path_model)
    data = pd.read_csv(data_path)
    X = data

    models = corr_run2(data=X,
                       data_cfg=cfg_user,
                       model_cfg=cfg_model)

    plt.plot(models[0].anomaly['score'])
    plt.plot(models[1].anomaly['score'])
    plt.plot(models[2].anomaly['score'])
    plt.plot(models[3].anomaly['score'])
    plt.show()

    plt.plot(models[4].anomaly['score'])
    plt.plot(models[5].anomaly['score'])
    plt.show()

    plt.plot(models[-1].anomaly['score'])
    plt.show()

    X['1_anom'] = models[0].anomaly['score']
    X['2_anom'] = models[1].anomaly['score']
    X['t_anom'] = models[-1].anomaly['score']

    print(1)
    # Collect results
    # f1 = features[0]
    # f1_model = features_models[f1]
    # f1_anomaly_scores = features_outputs[f1]['anomaly_score']
    # f1_anomaly_liklihoods = features_outputs[f1]['anomaly_likelihood']
    # f1_prediction_counts = features_outputs[f1]['pred_count']
