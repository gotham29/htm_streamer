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
               'permanenceInc': 0.1,
               'permanenceConnected': 0.3},
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
