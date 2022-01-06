import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import get_args, load_json, save_models, load_models, save_outputs
from source.config.config import load_config, save_config, build_enc_params, extend_features_samples
from source.model.model import init_models, run_models


def stream_to_htm(config_path, data_path):
    # 1. Load —> Config from Config Path
    cfg = load_config(config_path)

    # 2. Load —> ML Inputs from ML Inputs Path
    data = load_json(data_path)

    # 3. If Config[‘iter_current’] < Config[‘iter_samplesize’]:
    #     a. Store —> ML Inputs for Params
    if cfg['iter_current'] < cfg['iter_samplesize']:
        mode = 'sample_data'
        if cfg['iter_current'] == 0:
            cfg['features_samples'] = {f: [] for f in cfg['features']}
        else:
            cfg['features_samples'] = extend_features_samples(cfg['features_samples'], data)

    # 4. Elif cfg[‘iter_current’] == cfg[‘iter_samplesize’]:
    #     a. Store —> ML Inputs for Params
    #     b. Build —> Params
    #     c. Init —> Models
    #     d. Store —> Models
    elif cfg['iter_current'] == cfg['iter_samplesize']:
        mode = 'init_models'
        cfg['features_samples'] = extend_features_samples(cfg['features_samples'], data)
        cfg, features_enc_params = build_enc_params(cfg, cfg['features_samples'], cfg['models_encoders'])
        features_models = init_models(cfg['iter_current'],
                                      features_enc_params,
                                      cfg['models_predictor'],
                                      cfg['models_params'],
                                      cfg['models_for_each_feature'],
                                      cfg['models_encoders']['timestamp'])
        save_models(features_models, cfg['dir_models'])

    # 5. Else: (cfg['iter_current'] > cfg['iter_samplesize'])
    #     a. Load —> Models (from: cfg['dir_models'])
    #     b. Check —> if learn still true
    #     c. Run —> ML Inputs thru Models
    #     d. Store —> Models outputs (to: cfg['dir_results'])
    #     e. Store —> Models (to: cfg['dir_models'])
    else:
        mode = 'run_models'
        features_models = load_models(cfg['dir_models'])
        learn = True if cfg['iter_current'] < cfg['iter_stoplearn'] else False
        features_outputs = run_models(features_models, data, learn)
        save_outputs(features_outputs, cfg['dir_results'], cfg['iter_current'])
        save_models(features_models, cfg['dir_models'])

        if learn != cfg['learn']:
            print(f'  Learn changed!')
            print(f"      row = {cfg['iter_current']}")
            print(f"      {cfg['learn']} --> {learn}")
            cfg['learn'] = learn

    # 6. Update Config
    #     a. Check for —> mode change
    #     b. Increment —> iter_current
    if mode != cfg['mode']:
        print(f'  Mode changed!')
        print(f"      row = {cfg['iter_current']}")
        print(f"      {cfg['mode']} --> {mode}")
    cfg['mode'] = mode
    cfg['iter_current'] += 1

    # 7. Store —> Config
    save_config(cfg, config_path)


if __name__ == '__main__':
    args = get_args()
    stream_to_htm(args.config_path, args.data_path)
