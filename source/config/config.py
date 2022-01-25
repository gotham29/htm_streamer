import numpy as np
import os
import yaml


def load_config(yaml_path):
    """
    Purpose:
        Load config from path
    Inputs:
        yaml_path
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_config(cfg, yaml_path):
    """
    Purpose:
        Save config to path
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        yaml_path
            type: str
            meaning: .yaml path to save to
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- saved
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    return cfg


def build_enc_params(cfg, models_encoders, features_weights):
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


def get_rdse_resolution(feature, minmax, n_buckets):
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


def extend_features_samples(data, features_samples):
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


def get_default_params_htm():
    """
    Purpose:
        Provide default HTM model config -- as set in nupic.core/py/htm/example/hotgym.py
    Inputs:
        none
    Outputs:
        default_parameters
            type: dict
            meaning: 'models_params' (in config.yaml)
    """
    default_parameters = {
        'anomaly': {'period': 1000},
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
               'connectedPermanence': 0.5,
               'initialPerm': 0.21,
               'maxSegmentsPerCell': 128,
               'maxSynapsesPerSegment': 64,
               'minThreshold': 10,
               'newSynapseCount': 32,
               'permanenceDec': 0.1,
               'permanenceInc': 0.1},
    }
    return default_parameters


def get_default_params_predictor():
    """
    Purpose:
        Provide default config for htm.core Predictor
    Inputs:
        none
    Outputs:
        default_parameters
            type: dict
            meaning: 'models_predictor' (in config.yaml)
    """
    default_parameters = {
        'enable': False,
        'resolution': 1,
        'steps_ahead': [1, 2]
    }
    return default_parameters


def get_default_params_weights(features):
    """
    Purpose:
        Provide weight of 1.0 for each feature
    Inputs:
        none
    Outputs:
        default_parameters
            type: dict
            meaning: 'features_weights' (in config.yaml)
    """
    default_parameters = {f: 1.0 for f in features}
    return default_parameters


def get_default_params_encoder():
    """
    Purpose:
        Provide default config for htm.core RDSE
    Inputs:
        none
    Outputs:
        default_parameters
            type: dict
            meaning: 'models_encoders' (in config.yaml)
    """
    default_parameters = {
        'minmax_percentiles': [1, 99],
        'n': 700,
        'n_buckets': 140,
        'sparsity': 0.02,
        'timestamp': {
            'enable': False,
            'feature': 'timestamp',
            'timeOfDay': [30, 1],
            'weekend': 21
        },
    }
    return default_parameters


def get_mode(cfg):
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


def validate_params_timestep0(cfg):
    """
    Purpose:
        Add required params to config
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
    Outputs:
        cfg (params added)
    """
    # Add 'timesteps_stop' config
    if 'timesteps_stop' not in cfg:
        cfg['timesteps_stop'] = {}
    # Add 'learning' to 'timesteps_stop'
    if 'learning' not in cfg['timesteps_stop']:
        cfg['timesteps_stop']['learning'] = 1000000
    # Add 'timestep' & 'learn' to 'models_state'
    if 'timestep' not in cfg['models_state']:
        cfg['models_state']['timestep'] = 0
        cfg['models_state']['learn'] = False
    # Add 'track_tm' & 'track_iter' to 'models_state'
    if 'track_tm' not in cfg['models_state']:
        cfg['models_state']['track_tm'] = False
        cfg['models_state']['track_iter'] = 10
    # Ensure cfg['timesteps_stop']['sampling'] provided if 'features_minmax' ism't
    if 'features_minmax' not in cfg:
        assert 'timesteps_stop' in cfg, "'timesteps_stop' dict expected in config when 'features_minmax' not found"
        assert 'sampling' in cfg['timesteps_stop'], "'sampling' int expected in cfg['timesteps_stop']"

    return cfg


def validate_config(cfg, data, models_dir, outputs_dir):
    """
    Purpose:
        Ensure validity of all config values
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        data
            type: dict
            meaning: current data for each feature
        timestep
            type: int
            meaning: current timestep
        models_dir
            type: str
            meaning: path to dir where HTM models are written
        outputs_dir
            type: str
            meaning: path to dir where HTM outputs are written
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- validated & extended w/defaults
    """
    # Add params -- IF not found (first timestep)
    cfg = validate_params_timestep0(cfg)

    # Get mode
    cfg['models_state']['mode'] = get_mode(cfg)

    # Validate required params
    validate_params_required(cfg, data, models_dir, outputs_dir)

    # Assert starting models_states -- IF mode == 'sampling'
    if cfg['models_state']['mode'] == 'sampling':
        cfg['models_state']['learn'] = False

    # Assert valid params -- ONLY for INIT step
    elif cfg['models_state']['mode'] == 'initializing':
        cfg['models_state']['learn'] = True
        cfg = validate_params_init(cfg)

    else:  # Mode == 'running'
        timestep_current, timestep_init = cfg['models_state']['timestep'], cfg['models_state']['timestep_initialized']
        assert timestep_current > timestep_init, f"current timestep ({timestep_current}) <= timestep_initialized ({timestep_init})\n This shouldn't be when in 'running' mode"

    return cfg


def validate_params_required(cfg, data, models_dir, outputs_dir):
    """
    Purpose:
        Ensure valid entries for required config params
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        data
            type: dict
            meaning: current data for each feature
        models_dir
            type: str
            meaning: path to dir where HTM models are written
        outputs_dir
            type: str
            meaning: path to dir where HTM outputs are written
    Outputs:
        cfg (unchanged)
    """
    # Assert all expected params are present & correct type
    params_types = {
        'features': list,
        'models_state': dict,
        'timesteps_stop': dict,
    }
    for param, type in params_types.items():
        param_v = cfg[param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert timesteps_stop valid
    timesteps_stop_params_types = {
        k: int for k, v in cfg['timesteps_stop'].items()
    }
    for param, type in timesteps_stop_params_types.items():
        param_v = cfg['timesteps_stop'][param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert models_state valid
    modelsstate_params_types = {
        'learn': bool,
        'timestep': int,
        'track_tm': bool,
        'track_iter': int,
        'model_for_each_feature': bool,
    }
    for param, type in modelsstate_params_types.items():
        param_v = cfg['models_state'][param]
        assert isinstance(param_v, type), f"Param: {param} should be type {type}\n  Found --> {type(param_v)}"

    # Assert dirs exist
    for d in [models_dir, outputs_dir]:
        assert os.path.exists(d), f"dir not found --> {d}"

    # Assert features present in data:
    for f in cfg['features']:
        assert f in data, f"features missing from data --> {f}\n  Found --> {data.keys()}"

    # Assert timesteps_stop values valid
    if 'sampling' in cfg['timesteps_stop']:
        learning, sampling = cfg['timesteps_stop']['learning'], cfg['timesteps_stop']['sampling']
        assert learning > sampling, f"In 'timesteps_stop' config, expected 'learning' > 'sampling'" \
                                              f"but Found\n learning' = {learning}\n  " \
                                              f"'sampling' = {sampling}"


def validate_params_init(cfg):
    """
    Purpose:
        Ensure valid entries for config params -- used only in 'initializing' mode
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
    Outputs:
        cfg -- extended with all needed default params
    """
    # Get default model_params -- IF not provided
    if 'models_params' not in cfg:
        cfg['models_params'] = get_default_params_htm()

    # Get default models_predictor -- IF not provided
    if 'models_predictor' not in cfg:
        cfg['models_predictor'] = get_default_params_predictor()

    # Get default models_encoders -- IF not provided
    if 'models_encoders' not in cfg:
        cfg['models_encoders'] = get_default_params_encoder()

    if 'features_weights' not in cfg:
        cfg['features_weights'] = get_default_params_weights(cfg['features'])

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
