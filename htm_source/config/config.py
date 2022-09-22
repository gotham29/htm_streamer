import numpy as np


def reset_config(cfg: dict) -> dict:
    """
    Purpose:
        Reset config to minimal state
    Inputs:
        cfg:
            type: dict
            meaning: config to be reset
    Outputs:
        cfg:
            type: dict
            meaning: reset config
    """
    keys_keep = ['features', 'models_state', 'timesteps_stop']
    keys_keep_models_state = ['model_for_each_feature', 'use_sp']
    cfg = {k: v for k, v in cfg.items() if k in keys_keep}
    cfg['models_state'] = {k: v for k, v in cfg['models_state'].items()
                           if k in keys_keep_models_state}
    return cfg


def build_enc_params(cfg: dict,
                     models_encoders: dict,
                     features_weights: dict
                     ) -> (dict, dict):
    """
    Purpose:
        Set encoder params fpr each feature using sampled data
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        features_samples
            type: dict
            meaning: list of sampled values for each feature
        models_encoders
            type: dict
            meaning: encoder param values (user-specified in config.yaml)
    Outputs:
        cfg:
            type: dict
            meaning: config (yaml) -- updated to include RDSE resolutions for all features
        features_enc_params
            type: dict
            meaning: set of encoder params for each feature ('size, sparsity', 'resolution')
    """

    features_minmax = cfg.get('features_minmax', None)
    features_samples = cfg.get('features_samples', None)

    # if 'features_minmax' not user-provided, get from 'features_samples'
    if features_minmax is not None:
        pass
    else:
        min_perc, max_perc = models_encoders['minmax_percentiles']
        features_minmax = {feat: [np.percentile(sample, min_perc), np.percentile(sample, max_perc)] for feat, sample in
                           features_samples.items()}

    features_enc_params = {f: {} for f in features_minmax}
    cfg['features_minmax'] = {k: [str(v[0]), str(v[1])] for k, v in features_minmax.items()}
    cfg['features_resolutions'] = {}

    for f, minmax in features_minmax.items():
        features_enc_params[f]['size'] = int(models_encoders['n'] * features_weights[f])
        features_enc_params[f]['sparsity'] = models_encoders['sparsity']
        features_enc_params[f]['resolution'] = get_rdse_resolution(f,
                                                                   minmax,
                                                                   models_encoders['n_buckets'])
        cfg['features_resolutions'][f] = str(round(features_enc_params[f]['resolution'], 3))
    features_enc_params = {k: v for k, v in features_enc_params.items() if v['resolution'] != 0}
    return cfg, features_enc_params


def get_rdse_resolution(feature: str,
                        minmax: list,
                        n_buckets: int
                        ) -> float:
    """
    Purpose:
        Calculate 'resolution' pararm to RDSE for given feature
    Inputs:
        sample
            type: str
            meaning: name of given feature
        minmax
            type: list
            meaning: min & max feature values
        n_buckets
            type: int
            meaning: number of categories to divide (max-min) range over
    Outputs:
        resolution
            type: float
            meaning: param calculated for feature's RDSE encoder
    """
    minmax_range = float(minmax[1]) - float(minmax[0])
    resolution = minmax_range / float(n_buckets)
    if resolution == 0:
        print(f"Dropping feature, due to no variation in sample\n  --> {feature}")
    return resolution


def extend_features_samples(data: dict, features_samples: dict) -> dict:
    """
    Purpose:
        Add given data to feature's sample list
    Inputs:
        data
            type: dict
            meaning: current data for each feature
        features_samples
            type: dict
            meaning: list of samples collected for each feature so far
    Outputs:
        features_samples
            type: dict
            meaning: features_samples -- extemded to include current data
    """
    for f, sample in features_samples.items():
        sample.append(data[f])
    return features_samples


def get_mode(cfg: dict) -> str:
    """
    Purpose:
        Determine which mode to run ('sampling' / 'initializing' / 'running')
    Inputs:
        cfg:
            type: dict
            meaning: config yaml
    Outputs:
        mode:
            type: string
            meaning: which mode to use in current timestep
    """

    mode_prev = cfg['models_state'].get('mode', None)

    # sampling --> if: 'features_minmax' not in cfg
    if 'features_minmax' not in cfg:
        sampling_done = False if cfg['models_state']['timestep'] < cfg['timesteps_stop']['sampling'] else True
        mode = 'initializing'
        if not sampling_done:
            mode = 'sampling'

    else:  # sampling done
        # init --> if: 'timestep_initialized' doesn't exist yet
        if 'timestep_initialized' not in cfg['models_state']:  # models not built
            mode = 'initializing'
        # run --> otherwise
        else:  # models built
            mode = 'running'

    if mode_prev != mode:
        print(f'  Mode changed!')
        print(f"      row = {cfg['models_state']['timestep']}")
        print(f"      {mode_prev} --> {mode}")

    return mode
