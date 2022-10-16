import numpy as np

from htm_source.utils.general import isnumeric


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
    keys_keep_models_state = ['model_for_each_feature', 'use_sp', 'return_pred_count']
    cfg = {k: v for k, v in cfg.items() if k in keys_keep}
    cfg['models_state'] = {k: v for k, v in cfg['models_state'].items()
                           if k in keys_keep_models_state}
    return cfg


def get_params_rdse(f: str,
                    f_dict: dict,
                    models_encoders: dict,
                    f_weight: float,
                    f_sample: list) -> dict:
    """
    Purpose:
        Get enc params for 'f'
    Inputs:
        f:
            type: str
            meaning: name of feature
        f_dict:
            type: dict
            meaning: holds type, min & max for 'f'
        models_encoders:
            type: dict
            meaning: encoding params from config
        f_weight:
            type: float
            meaning: weight for 'f'
        f_sample:
            type: list
            meaning: values for 'f'
    Outputs:
        params_rdse
            type: dict
            meaning: enc params for 'f'
    """
    # use min/max if specified
    if isnumeric(f_dict['min']) and isnumeric(f_dict['max']):
        f_minmax = [f_dict['min'], f_dict['max']]
    # else find min/max from f_samples
    else:
        min_perc, max_perc = models_encoders['minmax_percentiles']
        f_minmax = [np.percentile(f_sample, min_perc), np.percentile(f_sample, max_perc)]
    params_rdse = {
        'size': int(models_encoders['n'] * f_weight),
        'sparsity': models_encoders['sparsity'],
        'resolution': get_rdse_resolution(f,
                                          f_minmax,
                                          models_encoders['n_buckets'])
    }
    return params_rdse


def build_enc_params(features: dict,
                     models_encoders: dict,
                     features_samples: dict,
                     types_numeric: list = ('int', 'float'),
                     types_time: list = ('timestamp', 'datetime')
                     ) -> dict:
    """
    Purpose:
        Set encoder params fpr each feature (using ether found or sampled min/max)
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
    features_weights = {k: v['weight'] for k, v in features.items()}
    features_enc_params = {}
    for f, f_dict in features.items():
        # get enc - numeric
        if f_dict['type'] in types_numeric:
            features_enc_params[f] = get_params_rdse(f,
                                                     f_dict,
                                                     models_encoders,
                                                     features_weights[f],
                                                     features_samples[f])
        # get enc - datetime
        elif f_dict['type'] in types_time:
            features_enc_params[f] = {k: v for k, v in f_dict.items() if k != 'type'}
        # get enc - categoric
        elif f_dict['type'] == 'category':
            raise NotImplementedError("Category encoder not implemented yet")
            # features_enc_params[f] = get_params_category(f_dict)
        else:
            raise TypeError(f"Unsupported type: {f_dict['type']}")
        features_enc_params[f]['type'] = f_dict['type']
    return features_enc_params


def get_rdse_resolution(feature: str,
                        minmax: list,
                        n_buckets: int
                        ) -> float:
    """
    Purpose:
        Calculate 'resolution' pararm to RDSE for given feature
    Inputs:
        feature
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
        print(f"Dropping feature, due to no variation in sample\n  --> {feature}")  # does this happen actually?
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

    if cfg['models_state']['timestep'] < cfg['timesteps_stop']['sampling']:
        mode = 'sampling'
    elif cfg['models_state']['timestep'] == cfg['timesteps_stop']['sampling']:
        mode = 'initializing'
    else:
        mode = 'running'

    if mode_prev != mode:
        print(f'  Mode changed!')
        print(f"      row = {cfg['models_state']['timestep']}")
        print(f"      {mode_prev} --> {mode}")

    return mode
