import numpy as np
import yaml


def load_config(yaml_path):
    """
    Load config from path
    Inputs:
        yaml_path: .yaml path to load from
    Outputs:
        cfg: dict
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_config(cfg, yaml_path):
    """
    Inputs:
        cfg: config from yaml (dict)
        yaml_path: .yaml path to save to (string)
    Outputs:
        cfg_out: config from yaml (dict)
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    return cfg


def build_enc_params(cfg, features_samples, models_encoders):
    features_enc_params = {f: {} for f in features_samples}
    cfg['models_encoders']['resolutions'] = {}
    for f, sample in features_samples.items():
        features_enc_params[f]['size'] = models_encoders['n']
        features_enc_params[f]['sparsity'] = models_encoders['sparsity']
        features_enc_params[f]['resolution'] = get_rdse_resolution(sample,
                                                                   models_encoders['minmax_percentiles'],
                                                                   models_encoders['n_buckets'])
        cfg['models_encoders']['resolutions'][f] = str(round(features_enc_params[f]['resolution'], 3))
    return cfg, features_enc_params


def get_rdse_resolution(sample, minmax_percentiles, n_buckets):
    min_perc, max_perc = minmax_percentiles[0], minmax_percentiles[1]
    f_min = np.percentile(sample, min_perc)
    f_max = np.percentile(sample, max_perc)
    minmax_range = f_max - f_min
    resolution = max(0.001, (minmax_range / float(n_buckets)))
    return resolution


def extend_features_samples(features_samples, data):
    for f, sample in features_samples.items():
        sample.append(data[f])
    return features_samples


def get_default_config():
    default_parameters = {
        # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
        'enc': {
            "value":
                {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
            "time":
                {'timeOfDay': (30, 1), 'weekend': 21}
        },
        'predictor': {'sdrc_alpha': 0.1},
        'sp': {'boostStrength': 3.0,
               'columnCount': 1638,
               'localAreaDensity': 0.04395604395604396,
               'potentialPct': 0.85,
               'synPermActiveInc': 0.04,
               'synPermConnected': 0.13999999999999999,
               'synPermInactiveDec': 0.006},
        'tm': {'activationThreshold': 17,
               'cellsPerColumn': 13,
               'initialPerm': 0.21,
               'maxSegmentsPerCell': 128,
               'maxSynapsesPerSegment': 64,
               'minThreshold': 10,
               'newSynapseCount': 32,
               'permanenceDec': 0.1,
               'permanenceInc': 0.1},
        'anomaly': {'period': 1000},
    }

    return default_parameters
