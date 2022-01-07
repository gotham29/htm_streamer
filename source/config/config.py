import numpy as np
import os
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


def extend_features_samples(data, features_samples):
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


def validate_config(cfg, data, timestep):
    # Assert all expected params are present & correct type
    params_types = {
        'dirs': dict,
        'features': list,
        'iters': dict,
        'models_state': dict,
        'models_encoders': dict,
        'models_params': dict,
        'models_predictor': dict,
    }
    for param, type in params_types.items():
        param_v = cfg[param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert iters valid
    iters_params_types = {
        'samplesize': int,
        'stop': int,
        'stoplearn': int
    }
    for param, type in iters_params_types.items():
        param_v = cfg['iters'][param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert models_state valid
    modelsstate_params_types = {
        'learn': bool,
        'mode': str,
        'timestep': int,
        'model_for_each_feature': bool,
    }
    for param, type in modelsstate_params_types.items():
        param_v = cfg['models_state'][param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert dirs exist
    for d in cfg['dirs']:
        assert os.path.exists(d), f"dir not found --> {d}"

    # Assert features present in data:
    for f in cfg['features']:
        assert f in data, f"features missing from data --> {f}\n  Found --> {data.keys()}"

    # Assert iter values valid
    assert cfg['iters']['stop'] > cfg['iters']['stoplearn'] > cfg['iters']['samplesize']

    # Assert starting models_states -- IF timestep=0
    if timestep == 0:
        cfg['models_state']['learn'] = True
        cfg['models_state']['mode'] = 'sample_data'

    # Assert valid params -- ONLY for INIT step
    if timestep == cfg['iters']['samplesize']:

        # Assert valid models_encoders dict
        enc_params_types = {
            'minmax_percentiles': list,
            'n': int,
            'n_buckets': int,
            'sparsity': float,
            'timestamp': dict,
        }
        for param, type in enc_params_types.items():
            param_v = cfg['models_encoders'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert minmax_percentiles valid
        min_perc = cfg['models_encoders']['minmax_percentiles'][0]
        max_perc = cfg['models_encoders']['minmax_percentiles'][1]
        assert min_perc < 10, f"Min percentile expected < 10\n  Found --> {min_perc}"
        assert max_perc > 90, f"Min percentile expected > 90\n  Found --> {max_perc}"

        # Assert n valid
        n = cfg['models_encoders']['n']
        assert n > 500, f"'n' should be > 500\n  Found --> {n}"

        # Assert n_buckets valid
        n_buckets = cfg['models_encoders']['n_buckets']
        assert n_buckets > 100, f"'n_buckets' should be > 100\n  Found --> {n_buckets}"

        # Assert sparsity valid
        sparsity = cfg['models_encoders']['sparsity']
        assert 0.01 < sparsity < 0.10, f"'sparsity' should be in range 0.01 - 0.10 \n  Found --> {sparsity}"

        # Assert valid timestamp dict
        timestamp_params_types = {
            'enable': bool,
            'feature': str,
            'timeOfDay': list,
            'weekend': int,
        }
        for param, type in timestamp_params_types.items():
            param_v = cfg['models_encoders']['timestamp'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid timestamp feature -- IF enabled
        if cfg['models_encoders']['timestamp']['enable']:
            time_feat = cfg['models_encoders']['timestamp']['feature']
            assert time_feat in data, f"time feature missing from data --> {time_feat}\n  Found --> {data.keys()}"

        # Assert valid timeOfDay
        ###

        # Assert valid weekend
        ###

        # Assert valid models_predictor
        predictor_params_types = {
            'enable': bool,
            'resolution': int,
            'steps_ahead': list,
        }
        for param, type in predictor_params_types.items():
            param_v = cfg['models_predictor'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid models_params
        model_params_types = {
            'anomaly': dict,
            'predictor': dict,
            'sp': dict,
            'tm': dict,
        }
        for param, type in model_params_types.items():
            param_v = cfg['models_params'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid anomaly_params_types
        anomaly_params_types = {
            'period': int,
        }
        for param, type in anomaly_params_types.items():
            param_v = cfg['models_params']['anomaly'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid predictor_params_types
        predictor_params_types = {
            'sdrc_alpha': float,
        }
        for param, type in predictor_params_types.items():
            param_v = cfg['models_params']['predictor'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid sp_params
        sp_params = {
            'boostStrength': float,
            'columnCount': int,
            'localAreaDensity': float,
            'potentialPct': float,
            'synPermActiveInc': float,
            'synPermConnected': float,
            'synPermInactiveDec': float,
        }
        for param, type in sp_params.items():
            param_v = cfg['models_params']['sp'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

        # Assert valid tm_params
        tm_params = {
            'activationThreshold': int,
            'cellsPerColumn': int,
            'initialPerm': float,
            'maxSegmentsPerCell': int,
            'maxSynapsesPerSegment': int,
            'minThreshold': int,
            'newSynapseCount': int,
            'permanenceDec': float,
            'permanenceInc': float,
        }
        for param, type in tm_params.items():
            param_v = cfg['models_params']['tm'][param]
            assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"
        print(f'\n  Config validated!')

    return cfg