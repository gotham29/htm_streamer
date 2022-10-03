def get_default_params_htm() -> dict:
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
        'sp': {'potentialPct': 0.8,
               'globalInhibition': True,
               'boostStrength': 0.0,
               'columnCount': 2048,
               'localAreaDensity': 0.0,
               'potentialPct': 0.85,
               'synPermActiveInc': 0.003,
               'synPermConnected': 0.2,
               'synPermInactiveDec': 0.0005,
               'numActiveColumnsPerInhArea': 40},
        'tm': {'activationThreshold': 20,
               'cellsPerColumn': 32,
               'columnDimensions': 2048,
               'initialPerm': 0.21,
               'maxSegmentsPerCell': 128,
               'maxSynapsesPerSegment': 128,
               'minThreshold': 13,
               'newSynapseCount': 32,
               'permanenceDec': 0.08,
               'permanenceInc': 0.04,
               'permanenceConnected': 0.5},
    }
    return default_parameters


def get_default_params_predictor() -> dict:
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


def get_default_params_weights(features: list) -> dict:
    """
    Purpose:
        Provide weight of 1.0 for each feature
    Inputs:
        features
            type: list
            meaning: features to assign equal weight to
    Outputs:
        default_parameters
            type: dict
            meaning: 'features_weights' (in config.yaml)
    """
    default_parameters = {f: 1.0 for f in features}
    return default_parameters


def get_default_params_encoder() -> dict:
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
        'n': 400,
        'w': 21,
        'n_buckets': 130,
        'p_padding': 20,
    }
    return default_parameters
