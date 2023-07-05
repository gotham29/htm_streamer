import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from htm_source.pipeline.htm_batch_runner import eyestate_run
from htm_source.utils.fs import load_config
import logging


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Load config & data
    config_path_user = os.path.join(os.getcwd(), 'config', 'config--eyestate.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = "/home/vlad/Theses/eye-state/eye-state.csv"
    cfg_user = load_config(config_path_user)
    cfg_model = load_config(config_path_model)

    data = pd.read_csv(data_path)
    X = data[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']]

    model = eyestate_run(data=X,
                         data_cfg=cfg_user,
                         model_cfg=cfg_model,
                         merge_mode='sd',
                         lazy_init=True)

    model.plot_results(data['eyeDetection'])

    print(1)
