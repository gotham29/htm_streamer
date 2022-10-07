import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from htm_source.utils import get_args
from htm_source.utils.fs import load_json, save_models, load_models, save_outputs, load_config, save_config
from htm_source.config import build_enc_params, extend_features_samples
from htm_source.config.validation import validate_config
from htm_source.model.runners import init_models, run_models, track_tm
from htm_source.pipeline.htm_batch_runner import run_batch


def stream_to_htm(config_path: str,
                  config_default_path,
                  data_path: str,
                  models_dir: str,
                  outputs_dir: str):
    """
    Purpose:
        Run HTM module -- in mode either: Sampling/Initializing/Running (depending on timestep)
    Inputs:
        config_path
            type: str
            meaning: path to config (yaml)
        data_path
            type: str
            meaning: path to stream data (json)
        models_dir
            type: str
            meaning: path to dir where HTM models are written to
        outputs_dir
            type: str
            meaning: path to dir where HTM outputs are written to
    Outputs:
        if mode == 'sample_data':
            stream data --> stored
        elif mode == 'init_models':
            stream data --> stored
            HTM models --> built
            HTM models --> stored
        else mode == 'run_models':
            HTM models --> loaded
            HTM models --> called on stream data
            HTM outputs --> stored
            HTM models --> stored
        config --> updated & stored
    """

    # 1. Load —> Config from Config Path
    cfg = load_config(config_path)
    cfg_default = load_config(config_default_path)

    # 2. Load —> ML Inputs from ML Inputs Path
    data = load_json(data_path)

    # 3. Validate --> Config
    cfg = validate_config(cfg=cfg,
                          cfg_default=cfg_default,
                          data=data,
                          models_dir=models_dir,
                          outputs_dir=outputs_dir)

    # 4. Mode == 'sampling'
    #     a. Store —> ML Inputs for Params
    if cfg['models_state']['mode'] == 'sampling':
        if cfg['models_state']['timestep'] == 0:
            cfg['features_samples'] = {f: [] for f in cfg['features']}
        else:
            cfg['features_samples'] = extend_features_samples(data=data,
                                                              features_samples=cfg['features_samples'])

    # 5. Mode == 'initializing':
    #     a. Build —> Params
    #     b. Init —> Models
    #     c. Train —> Models (on sample data)
    #     d. Store —> Models
    elif cfg['models_state']['mode'] == 'initializing':
        features_enc_params = build_enc_params(features=cfg['features'],
                                               features_samples=cfg['features_samples'],
                                               models_encoders=cfg['models_encoders'])
        features_models = init_models(use_sp=cfg['models_state']['use_sp'],
                                      return_pred_count=cfg['models_state']['return_pred_count'],
                                      models_params=cfg['models_params'],
                                      predictor_config=cfg['models_predictor'],
                                      features_enc_params=features_enc_params,
                                      model_for_each_feature=cfg['models_state']['model_for_each_feature'])
        features_models, features_outputs = run_batch(cfg=cfg,
                                                      cfg_default=cfg_default,
                                                      data=pd.DataFrame(cfg['features_samples']),
                                                      learn=True,
                                                      iter_print=100,
                                                      config_path=None,
                                                      config_default_path=None,
                                                      features_models=features_models)
        for f, outs in features_outputs.items():
            path_out = os.path.join(outputs_dir, f"sample--{f}.csv")
            pd.DataFrame(outs).to_csv(path_out)
        save_models(dir_models=models_dir,
                    features_models=features_models)
        cfg['models_state']['timestep_initialized'] = cfg['models_state']['timestep']

    # 6. Mode == 'running'
    #     a. Load —> Models (from: cfg['dirs']['models'])
    #     b. Check —> if learn still true
    #     c. Run —> ML Inputs thru Models
    #     d. Store —> Models TM state
    #     e. Store —> Models outputs (to: cfg['dirs']['results'])
    #     f. Store —> Models (to: cfg['dirs']['models'])
    else:
        features_models = load_models(models_dir)
        learn = True if cfg['models_state']['timestep'] < cfg['timesteps_stop']['learning'] else False
        features_outputs, features_models = run_models(learn=learn,
                                                       use_sp=cfg['models_state']['use_sp'],
                                                       features_data=data,
                                                       features_models=features_models,
                                                       timestep=cfg['models_state']['timestep'],
                                                       predictor_config=cfg['models_predictor'])
        if cfg['models_state']['track_tm']:
            is_track_timestep = cfg['models_state']['timestep'] % cfg['models_state']['track_iter'] == 0
            if is_track_timestep:
                print(f"  tracking TM, timestep --> {cfg['models_state']['timestep']}")
                cfg = track_tm(cfg, features_models)
        save_outputs(dir_out=outputs_dir,
                     timestep_init=cfg['models_state']['timestep_initialized'],
                     features_outputs=features_outputs,
                     timestep_current=cfg['models_state']['timestep'],
                     save_outputs_accumulated=cfg['models_state']['save_outputs_accumulated'])
        save_models(dir_models=models_dir,
                    features_models=features_models)

        if learn != cfg['models_state']['learn']:
            print(f'  Learn changed!')
            print(f"      row = {cfg['models_state']['timestep']}")
            print(f"      {cfg['models_state']['learn']} --> {learn}")
            cfg['models_state']['learn'] = learn

    # 7. Update Config
    #     a. Increment —> timestep
    cfg['models_state']['timestep'] += 1

    # 8. Store —> Config
    save_config(cfg=cfg,
                yaml_path=config_path)


if __name__ == '__main__':
    args = get_args()
    stream_to_htm(args.config_path, args.config_default_path, args.data_path, args.models_dir, args.outputs_dir)
