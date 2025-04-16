import os

import pandas as pd

from htm_streamer.pipeline.htm_batch_runner import run_batch
from htm_streamer.utils.fs import load_config

if __name__ == '__main__':
    # Specify run params
    model_for_each_feature = True
    return_pred_count = True
    features = ['3.3_Bus_Current',
                'Receiver_Doppler',
                'satellite_time']
    timestep_tostop_sampling = 40
    timestep_tostop_learning = 4000
    timestep_tostop_running = 5000

    # Load config & data
    config_path_user = os.path.join(os.getcwd(), 'tests', 'config--test.yaml')
    config_path_model = os.path.join(os.getcwd(), 'config', 'config--model_default.yaml')
    data_path = os.path.join(os.getcwd(), 'data', 'batch', 'sample_timeseries.csv')
    cfg_user = load_config(config_path_user)
    cfg_model = load_config(config_path_model)
    data = pd.read_csv(data_path)

    # Validate & update features
    cfg_features = {k: v for k, v in cfg_user['features'].items() if k in features}
    assert len(cfg_features) > 0, f"features don't match, check config.yaml\n  " \
                                  f"found: {features} \n  config.yaml: {cfg_user['features'].keys().to_list}"
    features_invalid = [f for f in cfg_features if f not in data]
    assert len(features_invalid) == 0, f"features not found --> {sorted(features_invalid)}"
    cfg_user['features'] = cfg_features
    cfg_user['timesteps_stop']['sampling'] = timestep_tostop_sampling
    cfg_user['timesteps_stop']['learning'] = timestep_tostop_learning
    cfg_user['timesteps_stop']['running'] = timestep_tostop_running
    cfg_user['models_state']['model_for_each_feature'] = model_for_each_feature
    cfg_user['models_state']['return_pred_count'] = return_pred_count

    # Train
    features_models, features_outputs = run_batch(cfg_user=cfg_user,
                                                  cfg_model=cfg_model,
                                                  config_path_user=None,
                                                  config_path_model=None,
                                                  learn=True,
                                                  data=data,
                                                  iter_print=100,
                                                  features_models={})

    # Run
    features_models, features_outputs = run_batch(cfg_user=cfg_user,
                                                  cfg_model=cfg_model,
                                                  config_path_user=None,
                                                  config_path_model=None,
                                                  learn=False,
                                                  data=data,
                                                  iter_print=100,
                                                  features_models=features_models)

    # Collect results
    f1 = features[0]
    f1_model = features_models[f1]
    f1_anomaly_scores = features_outputs[f1]['anomaly_score']
    f1_anomaly_liklihoods = features_outputs[f1]['anomaly_likelihood']
    f1_prediction_counts = features_outputs[f1]['pred_count']
