import os

import pandas as pd

from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.utils.fs import load_config

if __name__ == '__main__':
    # Load config & data
    config_path = os.path.join(os.getcwd(), 'data', 'config.yaml')
    data_path = os.path.join(os.getcwd(), 'data', 'batch', 'sample_timeseries.csv')
    config = load_config(config_path)
    data = pd.read_csv(data_path)

    my_features = data.columns[1:4].to_list()
    timestep_tostop_sampling = 40
    timestep_tostop_learning = 4000
    timestep_tostop_running = 5000

    model_for_each_feature = True
    features_invalid = [f for f in my_features if f not in data]

    assert len(features_invalid) == 0, f"features not found --> {sorted(features_invalid)}"

    config['features'] = my_features
    config['timesteps_stop']['sampling'] = timestep_tostop_sampling
    config['timesteps_stop']['learning'] = timestep_tostop_learning
    config['timesteps_stop']['running'] = timestep_tostop_running
    config['models_state']['model_for_each_feature'] = model_for_each_feature

    # Train
    features_models, features_outputs = run_batch(cfg=config,
                                                  config_path=None,
                                                  learn=True,
                                                  data=data,
                                                  iter_print=100,
                                                  features_models={})

    # Run
    features_models, features_outputs = run_batch(cfg=config,
                                                  config_path=None,
                                                  learn=False,
                                                  data=data,
                                                  iter_print=100,
                                                  features_models=features_models)

    # Collect results
    f1 = my_features[0]
    f1_model = features_models[f1]
    f1_anomaly_scores = features_outputs[f1]['anomaly_score']
    f1_anomaly_liklihoods = features_outputs[f1]['anomaly_likelihood']
    f1_prediction_counts = features_outputs[f1]['pred_count']
