import numpy as np
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor


class HTMModel():
    def __init__(self, iter_count, features_model, features_enc_params, models_params,
                 predictor_resolution, predictor_steps_ahead):
        self.iter_count = iter_count
        self.features_model = features_model
        self.features_enc_params = features_enc_params
        self.models_params = models_params
        self.predictor_resolution = predictor_resolution
        self.predictor_steps_ahead = predictor_steps_ahead
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

    def run(self, features_data, learn=True):
        # ENCODERS
        ## Call the encoders to create bit representations for each value.  These are SDR objects.
        encs_bits = [ SDR(0) ]
        for f in self.features_model:
            f_bits = self.features_encs[f].encode(features_data[f])
            encs_bits.append(f_bits)
        ## Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR(self.encoding_width).concatenate(encs_bits)

        # SPATIAL POOLER
        ## Create an SDR to represent active columns -- must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR(self.sp.getColumnDimensions())
        ## Execute Spatial Pooling algorithm over input space.
        self.sp.compute(encoding, learn, activeColumns)

        ## Get pred counts
        self.tm.activateDendrites(learn=False)
        pred_cells = self.tm.getPredictiveCells()
        n_pred_cells = pred_cells.getSum()
        n_cols_per_pred = round(self.models_params["sp"]["columnCount"] * self.models_params["sp"]["localAreaDensity"])
        pred_count = n_pred_cells / n_cols_per_pred

        # TEMPORAL MEMORY
        ## Execute Temporal Memory algorithm over active mini-columns.
        self.tm.compute(activeColumns, learn=learn)
        ## Get anomaly metrics
        anomaly_score = self.tm.anomaly
        anomaly_liklihood = self.anomaly_history.compute(anomaly_score)
        ## Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0:
            assert pred_count > 0, f"0 preds with anomaly={anomaly_score}"

        """
        # PREDICTOR
        ## Predict what will happen -- IF models_for_each_target==True
        targets_predictions = {}
        pdf = self.predictor.infer(self.tm.getActiveCells())
        for f in self.features_model:
            predictions = {}
            for step_ahead in self.predictor.steps:
                if pdf[step_ahead]:
                    predictions[step_ahead] = np.argmax(pdf[step_ahead]) * self.predictor_resolution
                else:
                    predictions[step_ahead] = np.nan
            # Train the predictor based on what just happened.
            self.predictor.learn(self.iter_count, self.tm.getActiveCells(), int(features_data[f] / self.predictor_resolution))
        """

        return anomaly_score, anomaly_liklihood, pred_count  # ,targets_predictions


def init_models(iter_count, features_enc_params, predictor_steps_ahead, predictor_resolution,
                models_params, models_for_each_feature):  # use_timestamp=False
    features_models = {}

    # if use_timestamp:
    #     # Make the Encoders.  These will convert input data into binary representations.
    #     dateEncoder = DateEncoder(timeOfDay= parameters["enc"]["time"]["timeOfDay"],
    #                               weekend  = parameters["enc"]["time"]["weekend"])
    #     encoders['timestamp'] = dateEncoder

    if models_for_each_feature:  # multiple models, one per feature
        for f in features_enc_params:
            features_model = [f]
            model = HTMModel(iter_count, features_model, features_enc_params, models_params,
                             predictor_resolution, predictor_steps_ahead)
            model.init_model()
            features_models[f] = model
            print(f'  model initialized --> {f}')
            print(f'    encoding_width = {model.encoding_width}')

    else:  # one multi-feature model
        features_model = list(features_enc_params.keys())
        model = HTMModel(iter_count, features_model, features_enc_params, models_params,
                         predictor_resolution, predictor_steps_ahead)
        model.init_model()
        features_models[f'megamodel_features={len(features_model)}'] = model
        print(f"  model initialized --> megamodel_features={len(features_model)}")
        print(f'    encoding_width = {model.encoding_width}')

    return features_models


def run_models(features_models, data, learn):
    features_outputs = {t:{} for t in features_models}
    for t, model in features_models.items():
        anomaly_score, anomaly_liklihood, pred_count = model.run(data, learn=learn)
        features_outputs[t] = {'anomaly_score':anomaly_score,
                              'anomaly_liklihood':anomaly_liklihood,
                              'pred_count':pred_count}
    return features_outputs