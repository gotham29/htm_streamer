import os
import sys

from htm_source.config import get_mode

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


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
    # Add 'sampling' to 'timesteps_stop'
    if 'sampling' not in cfg['timesteps_stop']:
        cfg['timesteps_stop']['sampling'] = 100
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
        cfg_user
            type: dict
            meaning: config with user-modified values ('features', 'models_state', 'timesteps_stop')
        cfg_model
            type: dict
            meaning: config with default values ('models_params', 'models_encoders', 'models_predictor')
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
        cfg
            type: dict
            meaning: config with combined user & default values
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
        if timestep_current <= timestep_init:
            msg = f"current timestep ({timestep_current}) <= timestep_initialized " \
                  f"({timestep_init})\n This shouldn't be when in 'running' mode"
            log.error(msg=msg)
            raise msg

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

    # Validate config param types
    ptypes_cfguser = {
        'features': dict,
        'models_state': dict,
        'timesteps_stop': dict,
    }
    ptypes_timesteps_stop = {k: int for k, v in cfg['timesteps_stop'].items()}
    ptypes_models_state = {
        'learn': bool,
        'timestep': int,
        'track_tm': bool,
        'use_sp': bool,
        'track_iter': int,
        'model_for_each_feature': bool,
    }
    ## cfg
    validate_param_types(cfg_types=ptypes_cfguser, cfg_values=cfg)
    ## cfg['timesteps_stop']
    validate_param_types(cfg_types=ptypes_timesteps_stop, cfg_values=cfg['timesteps_stop'])
    ## cfg['models_state']
    validate_param_types(cfg_types=ptypes_models_state, cfg_values=cfg['models_state'])

    # Assert dirs exist
    for d in [models_dir, outputs_dir]:
        if not os.path.exists(d):
            msg = f"dir not found --> {d}"
            log.error(msg=msg)
            raise msg

    # Assert features present in data:
    for f in cfg['features']:
        if not f in data:
            msg = f"features missing from data --> {f}\n  Found --> {data.keys()}"
            log.error(msg=msg)
            raise msg

    # Assert timesteps_stop values valid
    if 'learning' in cfg['timesteps_stop']:
        learning, sampling = cfg['timesteps_stop']['learning'], cfg['timesteps_stop']['sampling']
        if learning <= sampling:
            msg = f"In 'timesteps_stop' config, expected 'learning' > 'sampling'" \
                  f"but Found\n learning' = {learning}\n  " \
                  f"'sampling' = {sampling}"
            log.error(msg=msg)
            raise msg


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
    dicts_req = ['models_encoders', 'models_params', 'models_predictor', 'spatial_anomaly']
    for dreq in dicts_req:
        if dreq not in cfg:
            cfg[dreq] = cfg_model[dreq]

    # Validate config param types
    ptypes_model = {
        'anomaly_likelihood': dict,
        'sp': dict,
        'tm': dict,
    }
    ptypes_enc = {
        'n': int,
        'w': int,
        'n_buckets': int,
        'p_padding': int,
    }
    ptypes_predictor = {
        'enable': bool,
        'resolution': int,
        'steps_ahead': list,
        'sdrc_alpha': float
    }
    ptypes_anomlikelihood = {
        'probationaryPeriod': int,
        'reestimationPeriod': int,
    }
    ptypes_spatialanom = {
        'enable': bool,
        'tolerance': float,
        'perc_min': int,
        'perc_max': int,
        'anom_prop': float,
        'window': int,
    }
    ptypes_sp = {
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
    ptypes_tm = {
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

    ## cfg['models_params']
    validate_param_types(cfg_types=ptypes_model, cfg_values=cfg['models_params'])

    ## cfg['models_encoders']
    validate_param_types(cfg_types=ptypes_enc, cfg_values=cfg['models_encoders'])

    ## cfg['models_predictor']
    validate_param_types(cfg_types=ptypes_predictor, cfg_values=cfg['models_predictor'])

    ## cfg['spatial_anomaly']
    validate_param_types(cfg_types=ptypes_spatialanom, cfg_values=cfg['spatial_anomaly'])

    ## cfg['models_params']['anomaly_likelihood']
    validate_param_types(cfg_types=ptypes_anomlikelihood, cfg_values=cfg['models_params']['anomaly_likelihood'])

    ## cfg['models_params']['sp']
    validate_param_types(cfg_types=ptypes_sp, cfg_values=cfg['models_params']['sp'])

    ## cfg['models_params']['tm']
    validate_param_types(cfg_types=ptypes_tm, cfg_values=cfg['models_params']['tm'])

    error = False
    # Assert n valid
    n = cfg['models_encoders']['n']
    if n < 200:
        error = True
        msg = f"'n' should be > 200\n  Found --> {n}"

    # Assert w valid
    w = cfg['models_encoders']['w']
    if w < 0.05 * n:
        error = True
        msg = f"'w' should be > 5% of n \n  Found --> {w}\n  Should be at least --> {int(0.05 * n) + 1}"
    if w > 0.2 * n:
        error = True
        msg = f"'w' should be < 20% of n \n  Found --> {w}\n  Should be at most --> {int(0.2 * n) + 1}"

    # Assert n_buckets valid
    n_buckets = cfg['models_encoders']['n_buckets']
    if n_buckets < 100:
        error = True
        msg = f"'n_buckets' should be > 100\n  Found --> {n_buckets}"

    # Assert padding valid
    p_padding = cfg['models_encoders']['p_padding']
    if p_padding < -100:
        error = True
        msg = f"'p_padding' should be >= -100 \n  Found --> {p_padding}"
    if p_padding > 100:
        error = True
        msg = f"'p_padding' should be <= 100 \n  Found --> {p_padding}"

    if error:
        log.error(msg=msg)
        raise msg

    log.info(msg='  Config validated!')

    return cfg


def validate_param_types(cfg_types, cfg_values):
    for p, p_type in cfg_types.items():
        p_val = cfg_values[p]
        if not isinstance(p_val, p_type):
            msg = f"Param: {p} should be type {p_type}\n  Found --> {p_val} type:{type(p)}"
            log.error(msg=msg)
            raise msg

