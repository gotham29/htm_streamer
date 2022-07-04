import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE, RDSE_Parameters


class HTMmodel:
    def __init__(self, features_model, features_enc_params, models_params,
                 timestamp_config, predictor_config, use_sp):
        self.features_model = features_model
        self.features_enc_params = features_enc_params
        self.models_params = models_params
        self.timestamp_config = timestamp_config
        self.predictor_resolution = predictor_config['resolution']
        self.predictor_steps_ahead = predictor_config['steps_ahead']
        self.use_sp = use_sp
        self.sp = None
        self.tm = None
        self.predictor = None
        self.anomaly_history = None
        self.encoding_width = 0
        self.features_encs = {}

    def init_encs(self):
        """
        Purpose:
            Init HTMmodel encoders
        Inputs:
            HTMmodel.features_model
                type: list
                meaning: which data features to build HTMmodels for
            HTMmodel.features_enc_params
                type: dict
                meaning: encoder params built for each modeled feature
        Outputs:
            HTMmodel.features_encs
                type: dict
                meaning: HTMmodel encoder object for each feature modeled
            HTMmodel.encodingwidth
                type: int
                meaning: size in bits of total concatenated encoder (input to SP)
        """
        for f in self.features_model:
            scalar_encoder_params = RDSE_Parameters()
            scalar_encoder_params.size = self.features_enc_params[f]['size']
            scalar_encoder_params.sparsity = self.features_enc_params[f]["sparsity"]
            scalar_encoder_params.resolution = self.features_enc_params[f]['resolution']
            scalar_encoder = RDSE(scalar_encoder_params)
            self.encoding_width += scalar_encoder.size
            self.features_encs[f] = scalar_encoder
        # Add timestamp encoder -- if enabled
        if self.timestamp_config['enable']:
            date_encoder = DateEncoder(timeOfDay=self.timestamp_config["timeOfDay"],
                                       weekend=self.timestamp_config["weekend"])
            self.features_encs[self.timestamp_config['feature']] = date_encoder
            self.encoding_width += date_encoder.size

    def init_sp(self):
        """
        Purpose:
            Init HTMmodel.sp
        Inputs:
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
            HTMmodel.encoding_width
                type: int
                meaning: size in bits of total concatenated encoder (input to SP)
        Outputs:
            HTMmodel.sp
                type: htm.core.SpatialPooler
                meaning: HTM native alg that selects activeColumns for input to TM
        """
        sp_params = self.models_params["sp"]
        self.sp = SpatialPooler(
            inputDimensions=(self.encoding_width,),
            columnDimensions=(sp_params["columnCount"],),
            potentialPct=sp_params["potentialPct"],
            potentialRadius=self.encoding_width,
            globalInhibition=True,
            localAreaDensity=sp_params["localAreaDensity"],
            synPermInactiveDec=sp_params["synPermInactiveDec"],
            synPermActiveInc=sp_params["synPermActiveInc"],
            synPermConnected=sp_params["synPermConnected"],
            boostStrength=sp_params["boostStrength"],
            wrapAround=True
        )
        # sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

    def init_tm(self):
        """
        Purpose:
            Init HTMmodel.tm
        Inputs:
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            HTMmodel.tm
                type: htm.core.TemporalMemory
                meaning: HTM native alg that activates & depolarizes activeColumns' cells within HTM region
        """
        sp_params = self.models_params["sp"]
        tm_params = self.models_params["tm"]
        self.tm = TemporalMemory(
            columnDimensions=(sp_params["columnCount"],),
            cellsPerColumn=tm_params["cellsPerColumn"],
            activationThreshold=tm_params["activationThreshold"],
            initialPermanence=tm_params["initialPerm"],
            connectedPermanence=tm_params["permanenceConnected"],
            minThreshold=tm_params["minThreshold"],
            maxNewSynapseCount=tm_params["newSynapseCount"],
            permanenceIncrement=tm_params["permanenceInc"],
            permanenceDecrement=tm_params["permanenceDec"],
            predictedSegmentDecrement=0.0,
            maxSegmentsPerCell=tm_params["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tm_params["maxSynapsesPerSegment"]
        )
        # tm_info = Metrics( [tm.numberOfCells()], 999999999 )

    def init_anomalyhistory(self):
        """
        Purpose:
            Init HTMmodel.anomaly_history
        Inputs:
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            HTMmodel.anomaly_history
                type: htm.core.AnomalyLikelihood
                meaning: HTM native alg that yields normalized anomaly metric (likelihood) from window of anomaly scores
        """
        self.anomaly_history = AnomalyLikelihood(self.models_params["anomaly"]["period"])

    def init_predictor(self):
        """
        Purpose:
            Init HTMmodel.predictor
        Inputs:
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
            HTMmodel.predictor_steps_ahead
                type: list
                meaning: values for how many timestep(s) ahead to predict
        Outputs:
            HTMmodel.predictor
                type: htm.core.Predictor
                meaning: HTM native alg that yields predicted values for each step ahead -- in raw feature values
        """
        self.predictor = Predictor(steps=self.predictor_steps_ahead,
                                   alpha=self.models_params["predictor"]['sdrc_alpha'])

    def init_model(self):
        """
        Purpose:
            Init entire HTMmodel -- element by element
        Inputs:
            all data specified as input to consituent 'init_xx()' functions
        Outputs:
            HTMmodel
                type: HTMmodel
                meaning: Object that runs entire HTM process (enc+sp+tm+anomlikl+predictor) given raw data
        """
        self.init_encs()
        self.init_sp()
        self.init_tm()
        self.init_anomalyhistory()
        self.init_predictor()

    def get_encoding(self, features_data):
        """
        Purpose:
            Build total conccated encoding from all features' encoders -- for input to SP
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            HTMmodel.features_encs
                type: dict
                meaning: encoder objects for each feature
            HTMmodel.timestamp_config
                type: dict
                meaning: params for timestep encoding (user-provided in config.yaml)
            HTMmodel.encoding_width
                type: int
                meaning: size in bits of total concatenated encoder (input to SP)
        Outputs:
            encoding
                type: nup.core.SDR
                meaning: total concatenated encoding -- input to SP
        """
        encs_bits = [SDR(0)]
        # Get encodings for all features
        for f, enc in self.features_encs.items():
            ## Convert timestamp feature to datetime
            if f == self.timestamp_config['feature']:
                features_data[f] = pd.to_datetime(features_data[f])
            f_bits = enc.encode(features_data[f])
            encs_bits.append(f_bits)
        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(encs_bits)
        return encoding

    def get_predcount(self):
        """
        Purpose:
            Get number of predictions made by TM at current timestep
        Inputs:
            HTMmodel.tm
                type: nupic.core.TemporalMemory
                meaning: TemporalMemory component of HTMmodel
            HTMmodel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            pred_count
                type: float
                meaning: number of predictions made by TM at current timestep (# predicted cells / # active columns)
        """
        self.tm.activateDendrites(learn=False)
        pred_cells = self.tm.getPredictiveCells()
        # Count number of predicted cells
        n_pred_cells = pred_cells.getSum()
        n_cols_per_pred = round(self.models_params["sp"]["columnCount"] * self.models_params["sp"]["localAreaDensity"])
        # Normalize to number of predictions
        pred_count = n_pred_cells / n_cols_per_pred
        return pred_count

    def get_preds(self, timestep, f_data):
        """
        Purpose:
            Get predicted values for given feature -- for each of 'n_steps_ahead'
        Inputs:
            timestep
                type: int
                meaning: current timestep
            f_data
                type: float
                meaning: current value for given feature
            HTMmodel.predictor
                type: htm.core.Predictor
                meaning: HTM native alg that yields predicted values for each step ahead -- in raw feature values
            HTMmodel.tm
                type: nupic.core.TemporalMemory
                meaning: TemporalMemory component of HTMmodel
            HTMmodel.predictor_steps_ahead
                type: list
                meaning: set of steps ahead for HTMmodel.predictor
            HTMmodel.predictor_resolution
                type: int
                meaning: resolution param for HTMmodel.predictor
        Outputs:
            steps_predictions
                type: dict
                meaning: predicted feature values for each of 'n_steps_ahead'
        """
        pdf = self.predictor.infer(self.tm.getActiveCells())
        steps_predictions = {}
        # Get pred for each #/of steps ahead - IF available
        for step_ahead in self.predictor_steps_ahead:
            if pdf[step_ahead]:
                steps_predictions[step_ahead] = str(np.argmax(pdf[step_ahead]) * self.predictor_resolution)
            else:
                steps_predictions[step_ahead] = np.nan
        # Train the predictor based on what just happened.
        self.predictor.learn(timestep, self.tm.getActiveCells(),
                             int(f_data / self.predictor_resolution))
        return steps_predictions

    def run(self, features_data, timestep, learn, predictor_config):
        """
        Purpose:
            Run HTMmodel -- yielding all outputs & updating model (if 'learn'==True)
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            timestep
                type: int
                meaning: current timestep
            learn
                type: bool
                meaning: whether learning is enabled in HTMmodel
            predictor_config
                type: dict
                meaning: param values for HTMmodel.predictor (user-specified in config.yaml)
        Outputs:
            anomaly_score
                type: float
                meaning: anomaly metric - from HMTModel.tm
            anomaly_likelihood
                type: float
                meaning: anomaly metric - from HMTModel.anomaly_history
            pred_count
                type: float
                meaning: number of predictions made by TM at current timestep (# predicted cells / # active columns)
            steps_predictions
                type: dict
                meaning: predicted feature values for each of 'n_steps_ahead'
        """
        # ENCODERS
        # Call the encoders to create bit representations for each feature
        encoding = self.get_encoding(features_data)

        if self.use_sp:
            # SPATIAL POOLER
            # Create an SDR to represent active columns
            active_columns = SDR(self.sp.getColumnDimensions())
            self.sp.compute(encoding, learn, active_columns)
            print('    active_columns')
            print(f"      type --> {type(active_columns)}")
            print(f"      vals --> {active_columns}")
        else:
            active_columns = encoding.sparse  #np.where(encoding.sparse == 1)[0]

        # TEMPORAL MEMORY
        # Get prediction density
        pred_count = self.get_predcount()
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        anomaly_likelihood = self.anomaly_history.compute(anomaly_score)
        # Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0:
            assert pred_count > 0, f"0 preds with anomaly={anomaly_score}"

        # PREDICTOR
        # Predict raw feature value -- IF enabled AND model is 1 feature (excluding timestamp)
        steps_predictions = {}
        if predictor_config['enable'] and len(self.features_model) == 1:
            feature = self.features_model[0]
            steps_predictions = self.get_preds(timestep=timestep,
                                               f_data=features_data[feature])

        return anomaly_score, anomaly_likelihood, pred_count, steps_predictions


def init_models(features_enc_params, predictor_config,
                models_params, model_for_each_feature, timestamp_config, use_sp):
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
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
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
        for f in features_enc_params:
            features_model = [f]
            model = HTMmodel(features_model=features_model, features_enc_params=features_enc_params,
                             models_params=models_params, timestamp_config=timestamp_config,
                             predictor_config=predictor_config, use_sp=use_sp)
            model.init_model()
            features_models[f] = model

    else:  # one multi-feature model
        features_model = list(features_enc_params.keys())
        model = HTMmodel(features_model=features_model, features_enc_params=features_enc_params,
                         models_params=models_params, timestamp_config=timestamp_config,
                         predictor_config=predictor_config, use_sp=use_sp)
        model.init_model()
        features_models[f'megamodel_features={len(features_model)}'] = model

    print(f'  Models initialized...')
    for f, model in features_models.items():
        print(f'    {f}  (size={model.encoding_width})')

    return features_models


def run_models(timestep, features_data, learn, use_sp, features_models, timestamp_config, predictor_config):
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


def run_model(args):
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
    feature, HTMmodel, features_data, timestep, learn, predictor_config = args
    anomaly_score, anomaly_likelihood, pred_count, steps_predictions = HTMmodel.run(learn=learn,
                                                                                    timestep=timestep,
                                                                                    features_data=features_data,
                                                                                    predictor_config=predictor_config)
    result = {'model': HTMmodel,
              'feature': feature,
              'timestep': timestep,
              'pred_count': pred_count,
              'anomaly_score': anomaly_score,
              'steps_predictions': steps_predictions,
              'anomaly_likelihood': anomaly_likelihood}

    return result


def run_models_parallel(timestep, features_data, learn, use_sp, features_models, timestamp_config, predictor_config):
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
    max_workers = multiprocessing.cpu_count() - 1
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


def track_tm(cfg, features_models):
    # get TM state for each model
    features_tmstates = {f: {} for f in features_models}
    for feature, model in features_models.items():
        TemporalMemory = model.tm
        perm_connected = TemporalMemory.getConnectedPermanence()

        # get presynaptics (cells that are linked to) for each each
        cells = [_ for _ in range(TemporalMemory.connections.numCells())]
        cells_presynaptics = {c: TemporalMemory.connections.synapsesForPresynapticCell(c) for c in cells}

        # split synapses into 'potential' & 'formed'
        cells_potential, cells_formed = {}, {}
        total_synapses_potential, total_synapses_formed = 0, 0
        for cell, presynaptics in cells_presynaptics.items():
            if len(presynaptics) == 0:
                continue
            presyns_perms = {p: TemporalMemory.connections.permanenceForSynapse(p) for p in presynaptics}
            for presyn, perm in presyns_perms.items():
                # perm --> potential
                total_synapses_potential += 1
                if perm < perm_connected:
                    try:
                        cells_potential[cell].append(presyn)
                    except:
                        cells_potential[cell] = [presyn]
                # perm --> formed
                else:
                    total_synapses_formed += 1
                    try:
                        cells_formed[cell].append(presyn)
                    except:
                        cells_formed[cell] = [presyn]

        features_tmstates[feature]['potential'] = cells_potential
        features_tmstates[feature]['formed'] = cells_formed
        features_tmstates[feature]['total_synapses_potential'] = total_synapses_potential
        features_tmstates[feature]['total_synapses_formed'] = total_synapses_formed

    cfg['features_tmstates'] = features_tmstates

    return cfg
