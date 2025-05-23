import os
import sys

from htm_streamer.data.types import HTMType, to_htm_type
from htm_streamer.utils.general import isnumeric

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


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
                    f_min: float,
                    f_max: float,
                    # f_sample: list,
                    models_encoders: dict) -> dict:
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
        f_sample:
            type: list
            meaning: values for 'f'
        models_encoders:
            type: dict
            meaning: encoding params from config
    Outputs:
        params_rdse
            type: dict
            meaning: enc params for 'f'
    """
    # use min/max if specified
    if isnumeric(f_dict['min']) and isnumeric(f_dict['max']):
        f_min = f_dict['min']
        f_max = f_dict['max']
    # else find min/max from f_samples
    else:
        # f_min, f_max = min(f_sample), max(f_sample)
        rangePadding = abs(f_max - f_min) * (float(models_encoders['p_padding']) / 100)
        f_min = f_min - rangePadding
        f_max = f_max + rangePadding
    params_rdse = {
        'resolution': get_rdse_resolution(feature=f,
                                          f_min=f_min,
                                          f_max=f_max,
                                          n_buckets=models_encoders['n_buckets'])
    }
    return params_rdse


def get_params_category() -> dict:
    """
    Purpose:
        Get enc params for 'f'
    Inputs:
        n/a
    Outputs:
        params_category
            type: dict
            meaning: enc params for category
    """
    params_category = {'category': True}
    return params_category


def build_enc_params(features: dict,
                     models_encoders: dict,
                     ) -> dict:
    """
    Purpose:
        Set encoder params fpr each feature (using ether found or sampled min/max)
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        # features_samples
        #     type: dict
        #     meaning: list of sampled values for each feature
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
    features_weights = {k: v.get('weight', 1.0) for k, v in features.items()}
    features_enc_params = {}
    for f, f_dict in features.items():
        f_min = features[f]['min']
        f_max = features[f]['max']
        # get enc - numeric
        if to_htm_type(f_dict['type']) is HTMType.Numeric:
            features_enc_params[f] = get_params_rdse(f=f,
                                                     f_dict=f_dict,
                                                     f_min=f_min,
                                                     f_max=f_max,
                                                    #  f_sample=features_samples[f],
                                                     models_encoders=models_encoders, )
        # get enc - datetime
        elif to_htm_type(f_dict['type']) is HTMType.Datetime:
            features_enc_params[f] = {k: v for k, v in f_dict.items() if k != 'type'}
        # get enc - categoric
        elif to_htm_type(f_dict['type']) is HTMType.Categoric:
            features_enc_params[f] = get_params_category()
        else:
            raise TypeError(f"Unsupported type: {f_dict['type']}")
        # set seed, size, activeBits
        features_enc_params[f]['type'] = f_dict['type']
        features_enc_params[f]['seed'] = models_encoders['seed']
        features_enc_params[f]['size'] = int(models_encoders['n'] * features_weights[f])
        features_enc_params[f]['activeBits'] = int(models_encoders['w'] * features_weights[f])
    return features_enc_params


def get_rdse_resolution(feature: str,
                        f_min: float,
                        f_max: float,
                        n_buckets: int
                        ) -> float:
    """
    Purpose:
        Calculate 'resolution' pararm to RDSE for given feature
    Inputs:
        feature
            type: str
            meaning: name of given feature
        n_buckets
            type: int
            meaning: number of categories to divide (max-min) range over
    Outputs:
        resolution
            type: float
            meaning: param calculated for feature's RDSE encoder
    """
    resolution = (f_max - f_min) / float(n_buckets)
    if resolution == 0:
        resolution = 1.0

        log.info(msg=f"Dropping feature, due to no variation in sample\n  --> {feature}")  # for constants
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
        log.info(msg=f"  Mode changed @row {cfg['models_state']['timestep']}\n    {mode_prev} --> {mode}")

    return mode
