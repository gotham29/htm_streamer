import concurrent.futures
import multiprocessing as mp
from collections import defaultdict
from htm_source.model import HTMmodel


def init_models(use_sp: bool,
                models_params: dict,
                predictor_config: dict,
                features_enc_params: dict,
                model_for_each_feature: bool,
                types_time: list = ('timestamp', 'datetime'),
                ) -> dict:
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
        use_sp
            type: bool
            meaning: whether Spatial Pooler is enabled in HTMmodel
    Outputs:
        features_models
            type: dict
            meaning: HTMmodel for each feature
    """
    features_models = {}

    if model_for_each_feature:  # multiple models, one per feature
        params_time = {k: v for k, v in features_enc_params.items() if v['type'] in types_time}
        features_enc_params_numeric = {k: v for k, v in features_enc_params.items() if k not in params_time}
        for f in features_enc_params_numeric:
            params_f = {k: v for k, v in features_enc_params.items() if k == f}
            params_f = {**params_time, **params_f}
            model = HTMmodel(use_sp=use_sp,
                             models_params=models_params,
                             predictor_config=predictor_config,
                             features_enc_params=params_f)
            features_models[f] = model

    else:  # one multi-feature model
        model = HTMmodel(use_sp=use_sp,
                         models_params=models_params,
                         predictor_config=predictor_config,
                         features_enc_params=features_enc_params)
        features_models[f'megamodel_features={len(features_enc_params)}'] = model

    print(f'  Models initialized...')
    for f, model in features_models.items():
        print(f'    {f}  (size={model.encoding_width})')

    return features_models


def run_models(learn: bool,
               use_sp: bool,
               timestep: int,
               features_data: dict,
               features_models: dict,
               timestamp_config: dict,
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
        # add timestamp data -- IF feature present
        if timestamp_config['feature'] in features_data:
            time_feat = timestamp_config['feature']
            features_outputs[f][time_feat] = str(features_data[time_feat])

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
                                                                                     features_data=features_data,
                                                                                     predictor_config=predictor_config)
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
