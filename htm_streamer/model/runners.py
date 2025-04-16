import concurrent.futures
import multiprocessing as mp
import os
import sys
from collections import defaultdict

from htm_streamer.data import Feature, separate_time_and_rest
from htm_streamer.model import HTMmodel
from htm_streamer.utils import frozendict

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


def print_mod_params(mod):
    log.info('Model Params:')
    for k,v in mod.models_params.items():
        if k == 'sp' and not mod.use_sp:
            continue
        log.info(f"  {k}")
        for k_, v_ in v.items():
            log.info(msg=f"    {k_} = {v_}")


def init_models(use_sp: bool,
                return_pred_count: bool,
                models_params: dict,
                predictor_config: dict,
                features_enc_params: dict,
                spatial_anomaly_config: dict,
                model_for_each_feature: bool) -> dict:
    """
    Purpose:
        Build HTMmodels for each feature (features --> user-provided in config.yaml)
    Inputs:
        features_enc_params
            type: dict
            meaning: encoder params built for each modeled feature
        predictor_config
            type: dict
            meaning: param values for HTMmodel.predictor (user-specified in config.yaml)
        models_params
            type: dict
            meaning: HTM hyperparams (user-provided in config.yaml)
        model_for_each_feature
            type: bool
            meaning: whether to build a separate model for each feature (user-specified in config.yaml)
        return_pred_count
            type: bool
            meaning: whether to get the number of prediction along w/anomalies (causes slow-down)
        use_sp
            type: bool
            meaning: whether Spatial Pooler is enabled in HTMmodel
    Outputs:
        features_models
            type: dict
            meaning: HTMmodel for each feature
    """
    features_models = dict()
    features = {name: Feature(name, params) for name, params in features_enc_params.items()}
    if model_for_each_feature:  # multiple models, one per feature
        time_feature, non_time_features = separate_time_and_rest(features.values())
        for feat in non_time_features:
            single_feat = {time_feature: features[time_feature], feat: features[feat]}
            model = HTMmodel(features=frozendict(single_feat),
                             use_spatial_pooler=use_sp,
                             return_pred_count=return_pred_count,
                             models_params=models_params,
                             predictor_config=predictor_config,
                             spatial_anomaly_config=spatial_anomaly_config)
            # print_mod_params(model)
            features_models[feat] = model

    else:  # one multi-feature model
        model = HTMmodel(features=frozendict(features),
                         use_spatial_pooler=use_sp,
                         return_pred_count=return_pred_count,
                         models_params=models_params,
                         predictor_config=predictor_config,
                         spatial_anomaly_config=spatial_anomaly_config)
        # print_mod_params(model)
        features_models[f'megamodel_features={len(features_enc_params)}'] = model

    log.info(msg="  Models initialized...")
    for f, model in features_models.items():
        log.info(msg=f"    {f}")
        log.info(msg=f'      enc size = {model.encoding_width}')
        log.info(msg=f'      tm size  = {model.tm.numberOfColumns()}')

    return features_models


def run_models(learn: bool,
               use_sp: bool,
               timestep: int,
               features_data: dict,
               features_models: dict,
               predictor_config: dict
               ) -> (dict, dict):
    """
    Purpose:
        Update HTMmodel(s) & collect results for all features -- run in serial
    Inputs:
        timestep
            type: int
            meaning: current timestep
        features_data
            type: dict
            meaning: current timestep data for each feature
        learn
            type: bool
            meaning: whether learning is enabled in HTMmodel
        use_sp
            type: bool
            meaning: whether Spatial Pooler is enabled in HTMmodel
        features_models
            type: dict
            meaning: HTMmodel for each feature
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
        predictor_config
            type: dict
            meaning: param values for HTMmodel.predictor (user-specified in config.yaml)
    Outputs:
        features_outputs
            type: dict
            meaning: HTMmodel outputs for each feature
        features_models
            type: dict
            meaning: HTMmodels for each feature
    """
    features_outputs = {f: {} for f in features_models}
    # Get outputs & update models
    for f, model in features_models.items():
        args = (f, model, features_data, timestep, learn, use_sp, predictor_config)
        result = run_model(args)
        features_models[f] = result['model']
        features_outputs[f] = {k: v for k, v in result.items() if k not in ['model', 'feature']}
    return features_outputs, features_models


def run_model(args) -> dict:
    """
    Purpose:
        Update HTMmodel & collect result for 1 feature
    Inputs:
        timestep
            type: int
            meaning: current timestep
        features_data
            type: dict
            meaning: current timestep data for each feature
        learn
            type: bool
            meaning: whether learning is enabled in HTMmodel
        use_sp
            type: bool
            meaning: whether Spatial Pooler is enabled in HTMmodel
        features_models
            type: dict
            meaning: HTMmodel for each feature
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
        predictor_config
            type: dict
            meaning: param values for HTMmodel.predictor (user-specified in config.yaml)
    Outputs:
        result
            type: dict
            meaning: all outputs from HTMmodel.run() for given feature
    """
    feature, htm_model, features_data, timestep, learn, use_sp, predictor_config = args
    anomaly_score, anomaly_likelihood, pred_count, steps_predictions = htm_model.run(learn=learn,
                                                                                     timestep=timestep,
                                                                                     features_data=features_data)
    result = {'model': htm_model,
              'feature': feature,
              'timestep': timestep,
              'pred_count': pred_count,
              'anomaly_score': anomaly_score,
              'steps_predictions': steps_predictions,
              'anomaly_likelihood': anomaly_likelihood}

    return result


def run_models_parallel(timestep: int,
                        features_data: dict,
                        learn: bool,
                        use_sp: bool,
                        features_models: dict,
                        timestamp_config: dict,
                        predictor_config: dict
                        ) -> (dict, dict):
    """
    Purpose:
        Update HTMmodel(s) & collect results for all features -- run in parallel
    Inputs:
        timestep
            type: int
            meaning: current timestep
        features_data
            type: dict
            meaning: current timestep data for each feature
        learn
            type: bool
            meaning: whether learning is enabled in HTMmodel
        use_sp
            type: bool
            meaning: whether Spatial Pooler is enabled in HTMmodel
        features_models
            type: dict
            meaning: HTMmodel for each feature
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
        predictor_config
            type: dict
            meaning: param values for HTMmodel.predictor (user-specified in config.yaml)
    Outputs:
        features_outputs
            type: dict
            meaning: HTMmodel outputs for each feature
        features_models
            type: dict
            meaning: HTMmodels for each feature
    """
    features_outputs = {}
    models = []
    features = []
    models_count = len(features_models)
    learns = [learn for _ in range(models_count)]
    use_sps = [use_sp for _ in range(models_count)]
    timesteps = [timestep for _ in range(models_count)]
    features_datas = [features_data for _ in range(models_count)]
    predictor_configs = [predictor_config for _ in range(models_count)]

    for f, model in features_models.items():
        features.append(f)
        models.append(model)

    tasks = list(zip(features, models, features_datas, timesteps, learns, use_sps, predictor_configs))
    max_workers = mp.cpu_count() - 1
    chunksize = round(len(tasks) / max_workers / 4)
    chunksize = max(chunksize, 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_model, tasks, chunksize=chunksize)

    for result in results:
        features_models[result['feature']] = result['model']
        features_outputs[result['feature']] = {
            'timestep': result['timestep'],
            'pred_count': result['pred_count'],
            'anomaly_score': result['anomaly_score'],
            'steps_predictions': result['steps_predictions'],
            'anomaly_likelihood': result['anomaly_likelihood'],
        }
        # add timestamp data -- IF feature present
        if timestamp_config['feature'] in features_data:
            time_feat = timestamp_config['feature']
            features_outputs[result['feature']][time_feat] = str(features_data[time_feat])

    return features_outputs, features_models


def track_tm(cfg: dict, features_models: dict) -> dict:
    # get TM state for each model
    features_tm_states = dict()
    for feature, model in features_models.items():
        TemporalMemory = model.tm
        perm_connected = TemporalMemory.getConnectedPermanence()

        # split synapses into 'potential' & 'formed'
        cells_formed = defaultdict(list)
        cells_potential = defaultdict(list)
        total_synapses_potential, total_synapses_formed = 0, 0
        for cell_idx in range(TemporalMemory.connections.numCells()):
            presynaptics = TemporalMemory.connections.synapsesForPresynapticCell(cell_idx)
            if not presynaptics:
                continue

            for presyn in presynaptics:
                perm = TemporalMemory.connections.permanenceForSynapse(presyn)
                # perm --> potential
                total_synapses_potential += 1
                if perm < perm_connected:
                    cells_potential[cell_idx].append(presyn)
                # perm --> formed
                else:
                    total_synapses_formed += 1
                    cells_formed[cell_idx].append(presyn)

        features_tm_states[feature] = {'potential': cells_potential,
                                       'formed': cells_formed,
                                       'total_synapses_potential': total_synapses_potential,
                                       'total_synapses_formed': total_synapses_formed}

    cfg['features_tmstates'] = features_tm_states

    return cfg
