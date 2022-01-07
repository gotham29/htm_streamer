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
    def __init__(self, iter_count, features_model, features_enc_params, models_params,
                 models_enc_timestamp, predictor_config):
        self.iter_count = iter_count
        self.features_model = features_model
        self.features_enc_params = features_enc_params
        self.models_params = models_params
        self.models_enc_timestamp = models_enc_timestamp
        self.predictor_resolution = predictor_config['resolution']
        self.predictor_steps_ahead = predictor_config['steps_ahead']
        self.sp = None
        self.tm = None
        self.predictor = None
        self.anomaly_history = None
        self.encoding_width = 0
        self.features_encs = {}

    def init_encs(self):
        for f in self.features_model:
            scalar_encoder_params = RDSE_Parameters()
            scalar_encoder_params.size = self.features_enc_params[f]['size']
            scalar_encoder_params.sparsity = self.features_enc_params[f]["sparsity"]
            scalar_encoder_params.resolution = self.features_enc_params[f]['resolution']
            scalar_encoder = RDSE(scalar_encoder_params)
            self.encoding_width += scalar_encoder.size
            self.features_encs[f] = scalar_encoder
        # Add timestamp encoder -- if enabled
        if self.models_enc_timestamp['enable']:
            date_encoder = DateEncoder(timeOfDay=self.models_enc_timestamp["timeOfDay"],
                                       weekend=self.models_enc_timestamp["weekend"])
            self.features_encs[self.models_enc_timestamp['feature']] = date_encoder
            self.encoding_width += date_encoder.size

    def init_sp(self):
        spParams = self.models_params["sp"]
        sp = SpatialPooler(
            inputDimensions=(self.encoding_width,),
            columnDimensions=(spParams["columnCount"],),
            potentialPct=spParams["potentialPct"],
            potentialRadius=self.encoding_width,
            globalInhibition=True,
            localAreaDensity=spParams["localAreaDensity"],
            synPermInactiveDec=spParams["synPermInactiveDec"],
            synPermActiveInc=spParams["synPermActiveInc"],
            synPermConnected=spParams["synPermConnected"],
            boostStrength=spParams["boostStrength"],
            wrapAround=True
        )
        # sp_info = Metrics( sp.getColumnDimensions(), 999999999 )
        self.sp = sp

    def init_tm(self):
        sp_params = self.models_params["sp"]
        tm_params = self.models_params["tm"]
        tm = TemporalMemory(
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
        self.tm = tm

    def init_anomalyhistory(self):
        self.anomaly_history = AnomalyLikelihood(self.models_params["anomaly"]["period"])

    def init_predictor(self):
        self.predictor = Predictor(steps=self.predictor_steps_ahead,
                                   alpha=self.models_params["predictor"]['sdrc_alpha'])

    def init_model(self):
        self.init_encs()
        self.init_sp()
        self.init_tm()
        self.init_anomalyhistory()
        self.init_predictor()

    def get_encoding(self, features_data):
        encs_bits = [SDR(0)]
        # Get encodings for all features
        for f, enc in self.features_encs.items():
            ## Convert timestamp feature to datetime
            if f == self.models_enc_timestamp['feature']:
                features_data[f] = pd.to_datetime(features_data[f])
            f_bits = enc.encode(features_data[f])
            encs_bits.append(f_bits)
        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(encs_bits)
        return encoding

    def get_predcount(self):
        self.tm.activateDendrites(learn=False)
        pred_cells = self.tm.getPredictiveCells()
        # Count number of predicted cells
        n_pred_cells = pred_cells.getSum()
        n_cols_per_pred = round(self.models_params["sp"]["columnCount"] * self.models_params["sp"]["localAreaDensity"])
        # Normalize to number of predictions
        pred_count = n_pred_cells / n_cols_per_pred
        return pred_count

    def get_preds(self, features_data):
        feat = self.features_model[0]
        pdf = self.predictor.infer(self.tm.getActiveCells())
        steps_predictions = {}
        # Get pred for each #/of steps ahead - IF available
        for step_ahead in self.predictor_steps_ahead:
            if pdf[step_ahead]:
                steps_predictions[step_ahead] = np.argmax(pdf[step_ahead]) * self.predictor_resolution
            else:
                steps_predictions[step_ahead] = np.nan
        # Train the predictor based on what just happened.
        self.predictor.learn(self.iter_count, self.tm.getActiveCells(),
                             int(features_data[feat] / self.predictor_resolution))
        return steps_predictions

    def run(self, features_data, learn=True):
        # ENCODERS
        # Call the encoders to create bit representations for each feature
        encoding = self.get_encoding(features_data)

        # SPATIAL POOLER
        # Create an SDR to represent active columns
        active_columns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(encoding, learn, active_columns)

        # TEMPORAL MEMORY
        # Measure Prediction Density
        pred_count = self.get_predcount()
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        anomaly_liklihood = self.anomaly_history.compute(anomaly_score)
        # Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0:
            assert pred_count > 0, f"0 preds with anomaly={anomaly_score}"

        # PREDICTOR
        # Predict raw feature value -- IF model is 1 feature (excluding timestamp)
        steps_predictions = {}
        if len(self.features_model) == 1:
            steps_predictions = self.get_preds(features_data)

        return anomaly_score, anomaly_liklihood, pred_count, steps_predictions


def init_models(iter_count, features_enc_params, predictor_config,
                models_params, models_for_each_feature, models_enc_timestamp):

    features_models = {}

    if models_for_each_feature:  # multiple models, one per feature
        for f in features_enc_params:
            features_model = [f]
            model = HTMModel(iter_count, features_model, features_enc_params, models_params,
                             models_enc_timestamp, predictor_config)
            model.init_model()
            features_models[f] = model
            print(f'  model initialized --> {f}')
            print(f'    encoding_width = {model.encoding_width}')

    else:  # one multi-feature model
        features_model = list(features_enc_params.keys())
        model = HTMModel(iter_count, features_model, features_enc_params, models_params,
                         models_enc_timestamp, predictor_config)
        model.init_model()
        features_models[f'megamodel_features={len(features_model)}'] = model
        print(f"  model initialized --> megamodel_features={len(features_model)}")
        print(f'    encoding_width = {model.encoding_width}')

    return features_models


def run_models(features_models, data, learn, timestamp_config):
    features_outputs = {t: {} for t in features_models}
    for t, model in features_models.items():
        anomaly_score, anomaly_likelihood, pred_count, steps_predictions = model.run(data, learn=learn)
        features_outputs[t] = {'anomaly_score': anomaly_score,
                               'anomaly_likelihood': anomaly_likelihood,
                               'pred_count': pred_count,
                               'steps_predictions': steps_predictions}
        # add timestamp data -- IF enabled
        if timestamp_config['enable']:
            time_feat = timestamp_config['feature']
            features_outputs[t][time_feat] = str(data[time_feat])
    return features_outputs
