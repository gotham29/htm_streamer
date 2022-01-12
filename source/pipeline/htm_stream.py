import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import get_args, load_json, save_models, load_models, save_outputs
from source.config.config import load_config, save_config, build_enc_params, extend_features_samples, validate_config
from source.model.model import init_models, run_models


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
                          outputs_dir=outputs_dir,
                          timestep=cfg['models_state']['timestep'])

    # 4. If Config['models_state']['timestep'] < Config['timesteps_stop']['sampling']:
    #     a. Store —> ML Inputs for Params
    if cfg['models_state']['timestep'] < cfg['timesteps_stop']['sampling']:
        mode = 'sample_data'
        if cfg['models_state']['timestep'] == 0:
            cfg['features_samples'] = {f: [] for f in cfg['features']}
        else:
            cfg['features_samples'] = extend_features_samples(data=data,
                                                              features_samples=cfg['features_samples'])

    # 5. Elif cfg['models_state']['timestep'] == cfg['timesteps_stop']['sampling']:
    #     a. Store —> ML Inputs for Params
    #     b. Build —> Params
    #     c. Init —> Models
    #     d. Store —> Models
    elif cfg['models_state']['timestep'] == cfg['timesteps_stop']['sampling']:
        mode = 'init_models'
        cfg['features_samples'] = extend_features_samples(data=data,
                                                          features_samples=cfg['features_samples'])
        cfg, features_enc_params = build_enc_params(cfg=cfg,
                                                    features_samples=cfg['features_samples'],
                                                    models_encoders=cfg['models_encoders'])
        features_models = init_models(features_enc_params=features_enc_params,
                                      predictor_config=cfg['models_predictor'],
                                      models_params=cfg['models_params'],
                                      model_for_each_feature=cfg['models_state']['model_for_each_feature'],
                                      timestamp_config=cfg['models_encoders']['timestamp'])
        save_models(features_models=features_models,
                    dir_models=models_dir)

    # 6. Else: (cfg['models_state']['timestep'] > cfg['timesteps_stop']['sampling'])
    #     a. Load —> Models (from: cfg['dirs']['models'])
    #     b. Check —> if learn still true
    #     c. Run —> ML Inputs thru Models
    #     d. Store —> Models outputs (to: cfg['dirs']['results'])
    #     e. Store —> Models (to: cfg['dirs']['models'])
    else:
        mode = 'run_models'
        features_models = load_models(models_dir)
        learn = True if cfg['models_state']['timestep'] < cfg['timesteps_stop']['learning'] else False
        features_outputs = run_models(timestep=cfg['models_state']['timestep'],
                                      features_data=data,
                                      learn=learn,
                                      features_models=features_models,
                                      timestamp_config=cfg['models_encoders']['timestamp'],
                                      predictor_config=cfg['models_predictor'])
        save_outputs(features_outputs=features_outputs,
                     timestep_current=cfg['models_state']['timestep'],
                     timestep_sampling=cfg['timesteps_stop']['sampling'],
                     dir_out=outputs_dir)
        save_models(features_models=features_models,
                    dir_models=models_dir)

        if learn != cfg['models_state']['learn']:
            print(f'  Learn changed!')
            print(f"      row = {cfg['models_state']['timestep']}")
            print(f"      {cfg['models_state']['learn']} --> {learn}")
            cfg['models_state']['learn'] = learn

    # 7. Update Config
    #     a. Check for —> mode change
    #     b. Increment —> timestep
    if mode != cfg['models_state']['mode']:
        print(f'  Mode changed!')
        print(f"      row = {cfg['models_state']['timestep']}")
        print(f"      {cfg['models_state']['mode']} --> {mode}")
    cfg['models_state']['mode'] = mode
    cfg['models_state']['timestep'] += 1

    # 8. Store —> Config
    save_config(cfg=cfg,
                yaml_path=config_path)


if __name__ == '__main__':
    args = get_args()
    stream_to_htm(args.config_path, args.data_path, args.models_dir, args.outputs_dir)
