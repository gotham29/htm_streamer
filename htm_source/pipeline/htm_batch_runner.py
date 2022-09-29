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


def run_batch(cfg: Union[dict, None],
              config_path: Union[str, None],
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

    # 1. Load —> Config from Config Path IF not passed in as arg
    if cfg is None:
        cfg = load_config(config_path)
        print(f'\nLoaded —> Config from: {config_path}')

    # 2. Ensure --> Expected features present
    missing_feats = [f for f in cfg['features'] if f not in data]
    assert len(missing_feats) == 0, f"expected features missing!\n  --> {sorted(missing_feats)}"

    # 3. Init Models --> IF 'features_models' is empty
    do_init_models = True if len(features_models) == 0 else False
    if do_init_models:
        cfg['features_samples'] = {f: data[f].values for f in cfg['features']}
        cfg = validate_params_init(cfg)
        features_enc_params = build_enc_params(cfg=cfg,
                                               models_encoders=cfg['models_encoders'],
                                               features_weights=cfg['features_weights'])
        features_models = init_models(use_sp=cfg['models_state']['use_sp'],
                                      models_params=cfg['models_params'],
                                      predictor_config=cfg['models_predictor'],
                                      features_enc_params=features_enc_params,
                                      model_for_each_feature=cfg['models_state']['model_for_each_feature'])
        # save_models(dir_models=models_dir,
        #             features_models=features_models)
    try:
        timestep_limit = cfg['timesteps_stop']['running']
    except:
        timestep_limit = None

    # 4. Build --> 'features_outputs' data structure
    outputs_dict = {'anomaly_score': [], 'anomaly_likelihood': [], 'pred_count': []}
    # multi-models case
    if cfg['models_state']['model_for_each_feature']:
        features_outputs = {f: outputs_dict for f in cfg['features']}
    else:  # single-models case
        multi_feat = f"megamodel_features={len(cfg['features'])}"
        features_outputs = {multi_feat: outputs_dict}

    # 5. Run --> 'data' through 'features_models'
    print('Running main loop...')
    for timestep, row in data[:timestep_limit].iterrows():
        features_data = dict(row)

        # multi-models case
        if cfg['models_state']['model_for_each_feature']:
            for f, f_dict in cfg['features'].items():
                if to_htm_type(f_dict['type']) is HTMType.Datetime:
                    continue  # the rest of the code will never run!

                aScore, aLikl, pCount, sPreds = features_models[f].run(features_data, timestep, learn,
                                                                       cfg['models_predictor'])
                features_outputs[f]['anomaly_score'].append(aScore)
                features_outputs[f]['anomaly_likelihood'].append(aLikl)
                features_outputs[f]['pred_count'].append(pCount)
        else:  # single-models case
            aScore, aLikl, pCount, sPreds = features_models[multi_feat].run(features_data, timestep, learn,
                                                                            cfg['models_predictor'])
            features_outputs[multi_feat]['anomaly_score'].append(aScore)
            features_outputs[multi_feat]['anomaly_likelihood'].append(aLikl)
            features_outputs[multi_feat]['pred_count'].append(pCount)

        # report status
        if timestep > iter_print and timestep % iter_print == 0:
            print(f'  completed row: {timestep}')

    print(f"\nCONFIG['models_state']")
    print(f"  --> {cfg['models_state']}")

    return features_models, features_outputs


if __name__ == '__main__':
    args = get_args()
    features_models, features_outputs = run_batch(cfg=None,
                                                  data=pd.read_csv(args.data_path),
                                                  learn=True,
                                                  iter_print=100,
                                                  config_path=args.config_path,
                                                  features_models={})
