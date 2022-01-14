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


class HTMModel:
    def __init__(self, features_model, features_enc_params, models_params,
                 timestamp_config, predictor_config):
        self.features_model = features_model
        self.features_enc_params = features_enc_params
        self.models_params = models_params
        self.timestamp_config = timestamp_config
        self.predictor_resolution = predictor_config['resolution']
        self.predictor_steps_ahead = predictor_config['steps_ahead']
        self.sp = None
        self.tm = None
        self.predictor = None
        self.anomaly_history = None
        self.encoding_width = 0
        self.features_encs = {}

    def init_encs(self):
        """
        Purpose:
            Init HTMModel encoders
        Inputs:
            HTMModel.features_model
                type: list
                meaning: which data features to build HTMModels for
            HTMModel.features_enc_params
                type: dict
                meaning: encoder params built for each modeled feature
        Outputs:
            HTMModel.features_encs
                type: dict
                meaning: HTMModel encoder object for each feature modeled
            HTMModel.encodingwidth
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
            Init HTMModel.sp
        Inputs:
            HTMModel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
            HTMModel.encoding_width
                type: int
                meaning: size in bits of total concatenated encoder (input to SP)
        Outputs:
            HTMModel.sp
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
            Init HTMModel.tm
        Inputs:
            HTMModel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            HTMModel.tm
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
            connectedPermanence=sp_params["synPermConnected"],
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
            Init HTMModel.anomaly_history
        Inputs:
            HTMModel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
        Outputs:
            HTMModel.anomaly_history
                type: htm.core.AnomalyLikelihood
                meaning: HTM native alg that yields normalized anomaly metric (likelihood) from window of anomaly scores
        """
        self.anomaly_history = AnomalyLikelihood(self.models_params["anomaly"]["period"])

    def init_predictor(self):
        """
        Purpose:
            Init HTMModel.predictor
        Inputs:
            HTMModel.models_params
                type: dict
                meaning: HTM hyperparams (user-provided in config.yaml)
            HTMModel.predictor_steps_ahead
                type: list
                meaning: values for how many timestep(s) ahead to predict
        Outputs:
            HTMModel.predictor
                type: htm.core.Predictor
                meaning: HTM native alg that yields predicted values for each step ahead -- in raw feature values
        """
        self.predictor = Predictor(steps=self.predictor_steps_ahead,
                                   alpha=self.models_params["predictor"]['sdrc_alpha'])

    def init_model(self):
        """
        Purpose:
            Init entire HTMModel -- element by element
        Inputs:
            all data specified as input to consituent 'init_xx()' functions
        Outputs:
            HTMModel
                type: HTMModel
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
            HTMModel.features_encs
                type: dict
                meaning: encoder objects for each feature
            HTMModel.timestamp_config
                type: dict
                meaning: params for timestep encoding (user-provided in config.yaml)
            HTMModel.encoding_width
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
            HTMModel.tm
                type: nupic.core.TemporalMemory
                meaning: TemporalMemory component of HTMModel
            HTMModel.models_params
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
            HTMModel.predictor
                type: htm.core.Predictor
                meaning: HTM native alg that yields predicted values for each step ahead -- in raw feature values
            HTMModel.tm
                type: nupic.core.TemporalMemory
                meaning: TemporalMemory component of HTMModel
            HTMModel.predictor_steps_ahead
                type: list
                meaning: set of steps ahead for HTMModel.predictor
            HTMModel.predictor_resolution
                type: int
                meaning: resolution param for HTMModel.predictor
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
            Run HTMModel -- yielding all outputs & updating model (if 'learn'==True)
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            timestep
                type: int
                meaning: current timestep
            learn
                type: bool
                meaning: whether learning is enabled in HTMModel
            predictor_config
                type: dict
                meaning: param values for HTMModel.predictor (user-specified in config.yaml)
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

        # SPATIAL POOLER
        # Create an SDR to represent active columns
        active_columns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(encoding, learn, active_columns)

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
                models_params, model_for_each_feature, timestamp_config):
    """
    Purpose:
        Build HTMModels for each feature (features --> user-provided in config.yaml)
    Inputs:
        features_enc_params
            type: dict
            meaning: encoder params built for each modeled feature
        predictor_config
            type: dict
            meaning: param values for HTMModel.predictor (user-specified in config.yaml)
        models_params
            type: dict
            meaning: HTM hyperparams (user-provided in config.yaml)
        model_for_each_feature
            type: bool
            meaning: whether to build a separate model for each feature (user-specified in config.yaml)
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
    Outputs:
        features_models
            type: dict
            meaning: HTMModel for each feature
    """
    features_models = {}

    if model_for_each_feature:  # multiple models, one per feature
        for f in features_enc_params:
            features_model = [f]
            model = HTMModel(features_model, features_enc_params, models_params,
                             timestamp_config, predictor_config)
            model.init_model()
            features_models[f] = model

    else:  # one multi-feature model
        features_model = list(features_enc_params.keys())
        model = HTMModel(features_model, features_enc_params, models_params,
                         timestamp_config, predictor_config)
        model.init_model()
        features_models[f'megamodel_features={len(features_model)}'] = model

    print(f'  Models initialized...')
    for f, model in features_models.items():
        print(f'    {f}  (size={model.encoding_width})')

    return features_models


def run_models(timestep, features_data, learn, features_models, timestamp_config, predictor_config):
    """
    Purpose:
        Get HTMModel(s) outputs for all features
    Inputs:
        timestep
            type: int
            meaning: current timestep
        features_data
            type: dict
            meaning: current timestep data for each feature
        learn
            type: bool
            meaning: whether learning is enabled in HTMModel
        features_models
            type: dict
            meaning: HTMModel for each feature
        timestamp_config
            type: dict
            meaning: params for timestamp encoder (user-specified in config.yaml)
        predictor_config
            type: dict
            meaning: param values for HTMModel.predictor (user-specified in config.yaml)
    Outputs:
        features_outputs
            type: dict
            meaning: HTMModel outputs for each feature
    """
    features_outputs = {f: {} for f in features_models}
    # Get outputs for all features_model
    for f, model in features_models.items():
        anomaly_score, anomaly_likelihood, pred_count, steps_predictions = model.run(learn=learn,
                                                                                     timestep=timestep,
                                                                                     features_data=features_data,
                                                                                     predictor_config=predictor_config)
        features_models[f] = model
        features_outputs[f] = {'timestep': timestep,
                               'pred_count': pred_count,
                               'anomaly_score': anomaly_score,
                               'steps_predictions': steps_predictions,
                               'anomaly_likelihood': anomaly_likelihood}

        # args = (f, model, features_data, timestep, learn, predictor_config)
        # res = run_model(args)
        # features_models[f] = res['model']
        # features_outputs[f] = {'timestep': res['timestep'],
        #                        'pred_count': res['pred_count'],
        #                        'anomaly_score': res['anomaly_score'],
        #                        'steps_predictions': res['steps_predictions'],
        #                        'anomaly_likelihood': res['anomaly_likelihood']}

        # add timestamp data -- IF feature present
        if timestamp_config['feature'] in features_data:
            time_feat = timestamp_config['feature']
            features_outputs[f][time_feat] = str(features_data[time_feat])

    return features_outputs, features_models


def run_model(args):
    feature, HTMModel, features_data, timestep, learn, predictor_config = args
    anomaly_score, anomaly_likelihood, pred_count, steps_predictions = HTMModel.run(learn=learn,
                                                                                    timestep=timestep,
                                                                                    features_data=features_data,
                                                                                    predictor_config=predictor_config)
    out_dict = {'model': HTMModel,
                'feature': feature,
                'timestep': timestep,
                'pred_count': pred_count,
                'anomaly_score': anomaly_score,
                'steps_predictions': steps_predictions,
                'anomaly_likelihood': anomaly_likelihood}
    return out_dict



def run_models_parallel(timestep, features_data, learn, features_models, timestamp_config, predictor_config):

    features_outputs = {}
    models_count = len(features_models)
    models = []
    features = []
    learns = [learn for _ in range(models_count)]
    timesteps = [timestep for _ in range(models_count)]
    features_datas = [features_data for _ in range(models_count)]
    predictor_configs = [predictor_config for _ in range(models_count)]

    for f, model in features_models.items():
        features.append(f)
        models.append(model)

    tasks = list(zip(features, models, features_datas, timesteps, learns, predictor_configs))
    max_workers = multiprocessing.cpu_count() - 1
    chunksize = round(len(tasks) / max_workers / 4)
    chunksize = max(chunksize, 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_model, tasks, chunksize=chunksize)

    for res in results:
        features_models[res['feature']] = res['model']
        features_outputs[res['feature']] = {
            'timestep': res['timestep'],
            'pred_count': res['pred_count'],
            'anomaly_score': res['anomaly_score'],
            'steps_predictions': res['steps_predictions'],
            'anomaly_likelihood': res['anomaly_likelihood'],
        }
        # add timestamp data -- IF feature present
        if timestamp_config['feature'] in features_data:
            time_feat = timestamp_config['feature']
            features_outputs[res['feature']][time_feat] = str(features_data[time_feat])

    return features_outputs, features_models
