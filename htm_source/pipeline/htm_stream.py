import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from htm_source.utils.utils import get_args, load_json, save_models, load_models, save_outputs
from htm_source.config.config import load_config, save_config, build_enc_params, extend_features_samples, validate_config
from htm_source.model.model import init_models, run_models, run_models_parallel, track_tm


def stream_to_htm(config_path, data_path, models_dir, outputs_dir):
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

    # 2. Load —> ML Inputs from ML Inputs Path
    data = load_json(data_path)

    # 3. Validate --> Config
    cfg = validate_config(cfg=cfg,
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
    #     c. Store —> Models
    elif cfg['models_state']['mode'] == 'initializing':
        cfg, features_enc_params = build_enc_params(cfg=cfg,
                                                    models_encoders=cfg['models_encoders'],
                                                    features_weights=cfg['features_weights'])
        features_models = init_models(features_enc_params=features_enc_params,
                                      models_params=cfg['models_params'],
                                      predictor_config=cfg['models_predictor'],
                                      timestamp_config=cfg['models_encoders']['timestamp'],
                                      model_for_each_feature=cfg['models_state']['model_for_each_feature'],
                                      use_sp=cfg['models_state']['use_sp'])
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
                                                       features_data=data,
                                                       features_models=features_models,
                                                       timestep=cfg['models_state']['timestep'],
                                                       predictor_config=cfg['models_predictor'],
                                                       timestamp_config=cfg['models_encoders']['timestamp'])
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
    stream_to_htm(args.config_path, args.data_path, args.models_dir, args.outputs_dir)
