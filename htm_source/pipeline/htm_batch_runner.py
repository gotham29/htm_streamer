import os
import sys
from typing import Union

import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from htm_source.utils import get_args
from htm_source.utils.fs import load_config
from htm_source.config import build_enc_params
from htm_source.config.validation import validate_params_init
from htm_source.model.runners import init_models
from htm_source.data.types import HTMType, to_htm_type

from logger import setup_applevel_logger
log = setup_applevel_logger(file_name= os.path.join(_SOURCE_DIR,'logs.log') )


def run_batch(cfg_user: Union[dict, None],
              cfg_model: Union[dict, None],
              config_path_user: Union[str, None],
              config_path_model: Union[str, None],
              learn: bool,
              data: pd.DataFrame,
              iter_print: int,
              features_models: dict) -> (dict, dict):
    """
    Purpose:
        Loop over all rows in batch csv
            generate inputs for 'stream_to_htm'
            run 'stream_to_htm'
    Inputs:
        config_path
            type: str
            meaning: path to config (yaml)
        learn
            type: bool
            meaning: whether learning is on for features_models
        data
            type: dataframe
            meaning: data to run through models
        iter_print
            type: int
            meaning: how often to print status
        features_models
            type: dict
            meaning: existing models for each feature (if an
    Outputs:
        stream datas --> written to 'data_stream_dir'
        'stream_to_htm' --> called for each stream data
        'config.yaml' --> reset after last stream data
    """
    log.info("\n\n** run_batch()")

    # 1. Load —> Config from Config Path IF not passed in as arg
    if cfg_user is None:
        cfg_user = load_config(config_path_user)
        log.info(msg=f"  Loaded —> Config-User from: {config_path_user}")
    if cfg_model is None:
        cfg_model = load_config(config_path_model)
        log.info(msg=f"  Loaded —> Config-Model from: {config_path_model}")

    # 2. Ensure --> Expected features present
    missing_feats = [f for f in cfg_user['features'] if f not in data]
    assert len(missing_feats) == 0, f"  expected features missing!\n  --> {sorted(missing_feats)}"

    # 3. Init Models --> IF 'features_models' is empty
    do_init_models = True if len(features_models) == 0 else False
    if do_init_models:
        cfg_user['features_samples'] = {f: data[f].values for f in cfg_user['features']}
        cfg = validate_params_init(cfg_user, cfg_model)
        features_enc_params = build_enc_params(features=cfg['features'],
                                               features_samples=cfg['features_samples'],
                                               models_encoders=cfg['models_encoders'])
        features_models = init_models(use_sp=cfg['models_state']['use_sp'],
                                      return_pred_count=cfg['models_state']['return_pred_count'],
                                      models_params=cfg['models_params'],
                                      predictor_config=cfg['models_predictor'],
                                      features_enc_params=features_enc_params,
                                      spatial_anomaly_config=cfg['spatial_anomaly'],
                                      model_for_each_feature=cfg['models_state']['model_for_each_feature'])
        # save_models(dir_models=models_dir,
        #             features_models=features_models)
    timestep_limit = cfg_user['timesteps_stop'].get('running', None)

    # 4. Build --> 'features_outputs' data structure
    outputs_dict = {'anomaly_score': [], 'anomaly_likelihood': [], 'pred_count': []}
    # multi-models case
    if cfg_user['models_state']['model_for_each_feature']:
        features_outputs = {f: outputs_dict for f in cfg_user['features']}
    else:  # single-models case
        multi_feat = f"megamodel_features={len(cfg_user['features'])}"
        features_outputs = {multi_feat: outputs_dict}

    # 5. Run --> 'data' through 'features_models'
    log.info(msg="  Running main loop...")
    for timestep, row in data[:timestep_limit].iterrows():
        features_data = dict(row)
        # multi-models case
        if cfg_user['models_state']['model_for_each_feature']:
            for f, f_dict in cfg_user['features'].items():
                if to_htm_type(f_dict['type']) is HTMType.Datetime:
                    continue
                aScore, aLikl, pCount, sPreds = features_models[f].run(features_data, timestep, learn)
                features_outputs[f]['anomaly_score'].append(aScore)
                features_outputs[f]['anomaly_likelihood'].append(aLikl)
                features_outputs[f]['pred_count'].append(pCount)
        # single-models case
        else:
            aScore, aLikl, pCount, sPreds = features_models[multi_feat].run(features_data, timestep, learn)
            features_outputs[multi_feat]['anomaly_score'].append(aScore)
            features_outputs[multi_feat]['anomaly_likelihood'].append(aLikl)
            features_outputs[multi_feat]['pred_count'].append(pCount)
        # report status
        if timestep > iter_print and timestep % iter_print == 0:
            log.info(msg=f"  completed row: {timestep}")

    return features_models, features_outputs


if __name__ == '__main__':
    args = get_args()
    features_models, features_outputs = run_batch(cfg_user=None,
                                                  cfg_model=None,
                                                  data=pd.read_csv(args.data_path),
                                                  learn=True,
                                                  iter_print=100,
                                                  config_path_user=args.config_path_user,
                                                  config_path_model=args.config_path_model,
                                                  features_models={})
