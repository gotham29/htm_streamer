import math
from typing import Mapping, Union

capnp = None

import numpy as np
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, Predictor
from htm.bindings.sdr import SDR

# from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm_source.data import Feature, separate_time_and_rest, AnomalyLikelihood
from htm_source.utils import dict_zip, frozendict


class HTMmodel:
    def __init__(self,
                 features: frozendict[str, Feature],
                 models_params: dict,
                 predictor_config: dict,
                 use_sp: bool,
                 return_pred_count: bool = False):

        self.iteration_ = 0
        self.use_sp = use_sp
        self.features = features
        self.models_params = models_params
        self.predictor_resolution = predictor_config['resolution']
        self.predictor_steps_ahead = predictor_config['steps_ahead']
        self.return_pred_count = return_pred_count
        self.predictor = Predictor(steps=self.predictor_steps_ahead,
                                   alpha=self.models_params["predictor"]['sdrc_alpha'])
        self.encoding_width = sum(feat.encoding_size for feat in self.features.values())
        self.sp = self.init_sp()
        self.tm = self.init_tm()
        self.al = self.init_alikelihood()

        # utility attributes
        self.single_feature = self.get_single_feature_name()
        self.feature_names = list(self.features.keys())
        self.features_samples = {f: [] for f in self.feature_names}
        self.feature_timestamp = separate_time_and_rest(self.features.values())[0]

    def init_sp(self) -> Union[None, SpatialPooler]:
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
        if self.use_sp:
            return SpatialPooler(
                inputDimensions=(self.encoding_width,),
                columnDimensions=(self.models_params["sp"]["columnCount"],),
                potentialPct=self.models_params["sp"]["potentialPct"],
                potentialRadius=self.encoding_width,
                globalInhibition=self.models_params["sp"]["globalInhibition"],
                numActiveColumnsPerInhArea=self.models_params['sp']['numActiveColumnsPerInhArea'],
                synPermInactiveDec=self.models_params["sp"]["synPermInactiveDec"],
                synPermActiveInc=self.models_params["sp"]["synPermActiveInc"],
                synPermConnected=self.models_params["sp"]["synPermConnected"],
                boostStrength=self.models_params["sp"]["boostStrength"],
                localAreaDensity=self.models_params['sp']['localAreaDensity'],
                wrapAround=self.models_params['sp']['wrapAround'],
                seed=self.models_params['sp']['seed'])
        else:
            return None

    def init_tm(self) -> TemporalMemory:
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

        column_dimensions = self.models_params["sp"]["columnCount"] if self.use_sp else self.encoding_width
        return TemporalMemory(
            columnDimensions=(column_dimensions,),
            cellsPerColumn=self.models_params["tm"]["cellsPerColumn"],
            activationThreshold=self.models_params["tm"]["activationThreshold"],
            initialPermanence=self.models_params["tm"]["initialPerm"],
            connectedPermanence=self.models_params["tm"]["permanenceConnected"],
            minThreshold=self.models_params["tm"]["minThreshold"],
            maxNewSynapseCount=self.models_params["tm"]["newSynapseCount"],
            permanenceIncrement=self.models_params["tm"]["permanenceInc"],
            permanenceDecrement=self.models_params["tm"]["permanenceDec"],
            predictedSegmentDecrement=self.models_params["tm"]["predictedSegmentDecrement"],
            maxSegmentsPerCell=self.models_params["tm"]["maxSegmentsPerCell"],
            maxSynapsesPerSegment=self.models_params["tm"]["maxSynapsesPerSegment"],
            seed=self.models_params['sp']['seed'])

    def init_alikelihood(self) -> AnomalyLikelihood:
        LearningPeriod = int(math.floor(self.models_params["alikl"]["probationaryPeriod"] / 2.0))
        return AnomalyLikelihood(
            learningPeriod=LearningPeriod,
            estimationSamples=self.models_params["alikl"]["probationaryPeriod"] - LearningPeriod,
            reestimationPeriod=self.models_params["alikl"]["reestimationPeriod"])

    def get_alikelihood(self, value, anomaly_score, timestamp) -> float:
        anomalyScore = self.al.anomalyProbability(value, anomaly_score, timestamp)
        logScore = self.al.computeLogLikelihood(anomalyScore)
        return logScore

    def get_single_feature_name(self) -> Union[None, str]:
        """
        If the model has a single feature beside the timestamp, will return the name of that feature.
        Otherwise, returns None.
        """
        _, non_time_feature_names = separate_time_and_rest(self.features.values())
        if len(non_time_feature_names) == 1:
            return non_time_feature_names[0]
        else:
            return None

    def get_encoding(self, features_data: Mapping) -> SDR:
        """
        Purpose:
            Build total concatenated encoding from all features' encoders -- for input to SP
        Inputs:
            features_data
                type: dict
                meaning: current data for each feature
            HTMmodel.feature_encoders
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

        # Get encodings for all features
        all_encodings = [SDR(0)] + [feature.encode(data) for _, data, feature in dict_zip(features_data, self.features)]

        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(all_encodings)
        return encoding

    def get_predcount(self) -> float:
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

        # Count number of predicted cells
        n_pred_cells = self.tm.getPredictiveCells().getSum()
        n_cols_per_pred = self.tm.getWinnerCells().getSum()

        # Normalize to number of predictions
        pred_count = 0 if n_cols_per_pred == 0 else float(n_pred_cells) / n_cols_per_pred
        return pred_count

    def get_preds(self, timestep: int, f_data: float) -> dict:
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
        active_cells = self.tm.getActiveCells()
        pdf = self.predictor.infer(active_cells)
        steps_predictions = {}
        # Get pred for each #/of steps ahead - IF available
        for step_ahead in self.predictor_steps_ahead:
            if pdf[step_ahead]:
                steps_predictions[step_ahead] = np.argmax(pdf[step_ahead]) * self.predictor_resolution
            else:
                steps_predictions[step_ahead] = np.nan

        # Train the predictor based on what just happened.
        self.predictor.learn(timestep, active_cells, f_data // self.predictor_resolution)
        return steps_predictions

    def check_spatial_anomaly(self, params, features_data) -> bool:
        spatialAnomaly = False
        # filter time feature from features_data
        features_data = {f: val for f, val in features_data.items() if f != self.feature_timestamp}
        # get number of feats req for spatial anomaly
        anom_feats_req = max(int(len(features_data)*params['anom_prop']), 1)
        # update sample w/latest data
        for f, val in features_data.items():
            self.features_samples[f].append(val)
            self.features_samples[f] = self.features_samples[f][-params['window']:]
        # get max/min values (by percentile)
        features_minmax = {f: {} for f in features_data}
        for f in features_data:
            features_minmax[f]['min'] = np.percentile(self.features_samples[f], params['perc_min'])
            features_minmax[f]['max'] = np.percentile(self.features_samples[f], params['perc_max'])
        # get anom features
        features_anom = []
        for f, val in features_data.items():
            if val > features_minmax[f]['max'] or val < features_minmax[f]['min']:
                features_anom.append(f)
        # check for spatial anom
        if len(features_anom) >= anom_feats_req:
            spatialAnomaly = True
        return spatialAnomaly

    def run(self,
            features_data: Mapping,
            timestep: int,
            learn: bool,
            predictor_config: dict
            ) -> (float, float, float, dict):
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
        # select only relevant features
        features_data = {name: features_data[name] for name in self.feature_names}

        # ENCODERS
        # Call the encoders to create bit representations for each feature
        encoding = self.get_encoding(features_data)

        # SPATIAL POOLER (or just encoding)
        if self.use_sp:
            # Create an SDR to represent active columns
            active_columns = SDR(self.sp.getColumnDimensions())
            self.sp.compute(encoding, learn, active_columns)
        else:
            active_columns = encoding

        # TEMPORAL MEMORY
        # Get prediction density
        pred_count = self.get_predcount() if self.return_pred_count else None
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        # Choose feature for value arg
        f1 = self.single_feature if self.single_feature else list(features_data.keys())[0]
        # Get timestamp data if available
        timestamp = self.iteration_ if not self.feature_timestamp else features_data[self.feature_timestamp]
        anomaly_likelihood = self.get_alikelihood(value=features_data[f1],  # TODO - resolve megamodel case
                                                  timestamp=timestamp,
                                                  anomaly_score=anomaly_score)

        # Check for spatial anomaly (NAB)
        if self.models_params['spatial_anom']['enable']:
            spatialAnomaly = self.check_spatial_anomaly(self.models_params['spatial_anom'], features_data)
            if spatialAnomaly:
                anomaly_likelihood = 1.0

        # Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0 and pred_count == 0:
            raise RuntimeError(f"0 preds with anomaly={anomaly_score}")

        # PREDICTOR
        # Predict raw feature value -- IF enabled AND model is 1 feature (excluding timestamp)
        if self.single_feature and predictor_config['enable']:
            steps_predictions = self.get_preds(timestep=timestep,
                                               f_data=features_data[self.single_feature])
        else:
            steps_predictions = None

        # Increment iteration
        self.iteration_ += 1

        return anomaly_score, anomaly_likelihood, pred_count, steps_predictions
