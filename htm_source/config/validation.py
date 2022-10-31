import os

from htm_source.config import get_mode


def validate_params_timestep0(cfg: dict) -> dict:
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
        cfg['timesteps_stop']['learning'] = None 
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


def validate_config(cfg_user: dict,
                    cfg_model: dict,
                    data: dict,
                    models_dir: str,
                    outputs_dir: str
                    ) -> dict:
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
    cfg = validate_params_timestep0(cfg_user)

    # Get mode
    cfg['models_state']['mode'] = get_mode(cfg)

    # Set save_outputs_accumulated -- If not found
    if 'save_outputs_accumulated' not in cfg['models_state']:
        cfg['models_state']['save_outputs_accumulated'] = True

    # Validate required params
    validate_params_required(cfg, data, models_dir, outputs_dir)

    # Assert starting models_states -- IF mode == 'sampling'
    if cfg['models_state']['mode'] == 'sampling':
        cfg['models_state']['learn'] = False

    # Assert valid params -- ONLY for INIT step
    elif cfg['models_state']['mode'] == 'initializing':
        cfg['models_state']['learn'] = True
        cfg = validate_params_init(cfg, cfg_model)

    else:  # Mode == 'running'
        timestep_current, timestep_init = cfg['models_state']['timestep'], cfg['models_state']['timestep_initialized']
        assert timestep_current > timestep_init, f"current timestep ({timestep_current}) <= timestep_initialized " \
                                                 f"({timestep_init})\n This shouldn't be when in 'running' mode"

    return cfg


def validate_params_required(cfg: dict,
                             data: dict,
                             models_dir: str,
                             outputs_dir: str):
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
        'features': dict,
        'models_state': dict,
        'timesteps_stop': dict,
    }
    for param, p_type in params_types.items():
        param_v = cfg[param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert timesteps_stop valid
    timesteps_stop_params_types = {
        k: int for k, v in cfg['timesteps_stop'].items()
    }
    for param, p_type in timesteps_stop_params_types.items():
        param_v = cfg['timesteps_stop'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert models_state valid
    modelsstate_params_types = {
        'learn': bool,
        'timestep': int,
        'track_tm': bool,
        'use_sp': bool,
        'track_iter': int,
        'model_for_each_feature': bool,
    }
    for param, p_type in modelsstate_params_types.items():
        param_v = cfg['models_state'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

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


def validate_params_init(cfg: dict, cfg_model: dict) -> dict:
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
    ## Get cfg dicts not found in cfg
    dicts_req = ['models_encoders', 'models_params', 'models_predictor']
    for dreq in dicts_req:
        if dreq not in cfg:
            cfg[dreq] = cfg_model[dreq]

    # Assert valid models_encoders dict
    enc_params_types = {
        'n': int,
        'w': int,
        'n_buckets': int,
        'p_padding': int,
    }
    for param, p_type in enc_params_types.items():
        param_v = cfg['models_encoders'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert n valid
    n = cfg['models_encoders']['n']
    assert n > 200, f"'n' should be > 200\n  Found --> {n}"

    # Assert w valid
    w = cfg['models_encoders']['w']
    assert w >= 0.05*n, f"'w' should be > 5% of n \n  Found --> {w}\n  Should be at least --> {int(0.05*n)+1}"
    assert w <= 0.2*n, f"'w' should be < 20% of n \n  Found --> {w}\n  Should be at most --> {int(0.2*n)+1}"

    # Assert n_buckets valid
    n_buckets = cfg['models_encoders']['n_buckets']
    assert n_buckets > 100, f"'n_buckets' should be > 100\n  Found --> {n_buckets}"

    # Assert padding valid
    p_padding = cfg['models_encoders']['p_padding']
    assert p_padding >= -100, f"'p_padding' should be >= -100 \n  Found --> {p_padding}"
    assert p_padding <= 100, f"'p_padding' should be <= 100 \n  Found --> {p_padding}"

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
    for param, p_type in predictor_params_types.items():
        param_v = cfg['models_predictor'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert valid models_params
    model_params_types = {
        'anomaly_likelihood': dict,
        'predictor': dict,
        'sp': dict,
        'tm': dict,
    }
    for param, p_type in model_params_types.items():
        param_v = cfg['models_params'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert valid anomaly_params_types
    anomaly_params_types = {
        'probationaryPeriod': int,
        'reestimationPeriod': int,
    }
    for param, p_type in anomaly_params_types.items():
        param_v = cfg['models_params']['anomaly_likelihood'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert valid predictor_params_types
    predictor_params_types = {
        'sdrc_alpha': float,
    }
    for param, p_type in predictor_params_types.items():
        param_v = cfg['models_params']['predictor'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

    # Assert valid sp_params
    sp_params = {
        'globalInhibition': bool,
        'potentialPct': float,
        'potentialPct': float,
        'boostStrength': float,
        'synPermActiveInc': float,
        'synPermConnected': float,
        'localAreaDensity': float,
        'synPermInactiveDec': float,
        'columnCount': int,
        'numActiveColumnsPerInhArea': int,
    }

    for param, p_type in sp_params.items():
        param_v = cfg['models_params']['sp'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"

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
    for param, p_type in tm_params.items():
        param_v = cfg['models_params']['tm'][param]
        assert isinstance(param_v, p_type), f"Param: {param} should be type {p_type}\n  Found --> {p_type(param_v)}"
    print(f'  Config validated!')

    return cfg
