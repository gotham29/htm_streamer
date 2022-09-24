import numpy as np
import pandas as pd
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, Predictor
from htm.bindings.sdr import SDR
from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE_Parameters, RDSE


class HTMmodel:
    def __init__(self,
                 # features_model: list,
                 features_enc_params: dict,
                 models_params: dict,
                 # timestamp_config: dict,
                 predictor_config: dict,
                 use_sp: bool):
        # self.features_model = features_model
        self.features_enc_params = features_enc_params
        self.models_params = models_params
        # self.timestamp_config = timestamp_config
        self.predictor_resolution = predictor_config['resolution']
        self.predictor_steps_ahead = predictor_config['steps_ahead']
        self.use_sp = use_sp
        self.sp = None
        self.tm = None
        self.predictor = None
        self.anomaly_history = None
        self.encoding_width = 0
        self.features_encs = {}
        self.types_time = ['timestamp', 'datetime']
        self.types_numeric = ['int', 'float']

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
        for f, enc_params in self.features_enc_params.items():
            enc = self.get_encoder(enc_params)
            self.encoding_width += enc.size
            self.features_encs[f] = enc
        # for f in self.features_model:
        #     scalar_encoder_params = RDSE_Parameters()
        #     scalar_encoder_params.size = self.features_enc_params[f]['size']
        #     scalar_encoder_params.sparsity = self.features_enc_params[f]["sparsity"]
        #     scalar_encoder_params.resolution = self.features_enc_params[f]['resolution']
        #     scalar_encoder = RDSE(scalar_encoder_params)
        #     self.encoding_width += scalar_encoder.size
        #     self.features_encs[f] = scalar_encoder
        # # Add timestamp encoder -- if enabled
        # if self.timestamp_config['enable']:
        #     date_encoder = DateEncoder(timeOfDay=self.timestamp_config["timeOfDay"],
        #                                weekend=self.timestamp_config["weekend"])
        #     self.features_encs[self.timestamp_config['feature']] = date_encoder
        #     self.encoding_width += date_encoder.size

    def get_encoder(self, enc_params):
        """
        Purpose:
            Get initialized encoder object from params
        Inputs:
            enc_params
                type: dict
                meaning: params to init enc
        Outputs:
            enc:
                type: enc object (RDSE or DateEncoder)
        """
        if enc_params['type'] in self.types_numeric:
            rdse_params = RDSE_Parameters()
            rdse_params.size = enc_params['size']
            rdse_params.sparsity = enc_params["sparsity"]
            rdse_params.resolution = enc_params['resolution']
            enc = RDSE(rdse_params)
        elif enc_params['type'] in self.types_time:
            enc = DateEncoder(timeOfDay=enc_params["timeOfDay"],
                              weekend=enc_params["weekend"])
        else:
            raise NotImplementedError("Category encoder not implemented yet")

        return enc

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
        columnDimensions = sp_params["columnCount"] if self.use_sp else self.encoding_width

        self.tm = TemporalMemory(
            columnDimensions=(columnDimensions,),
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
        if self.use_sp:
            self.init_sp()
        self.init_tm()
        self.init_anomalyhistory()
        self.init_predictor()

    def get_encoding(self, features_data: dict) -> SDR:
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
            # Convert timestamp feature to datetime
            if self.features_enc_params[f]['type'] in self.types_time:  # f == self.timestamp_config['feature']:
                features_data[f] = pd.to_datetime(features_data[f])
            f_bits = enc.encode(features_data[f])
            encs_bits.append(f_bits)
        # Combine all features encodings into one for Spatial Pooling
        encoding = SDR(self.encoding_width).concatenate(encs_bits)
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
        pred_cells = self.tm.getPredictiveCells()
        # Count number of predicted cells
        n_pred_cells = pred_cells.getSum()
        n_cols_per_pred = round(self.models_params["sp"]["columnCount"] * self.models_params["sp"]["localAreaDensity"])
        # Normalize to number of predictions
        pred_count = n_pred_cells / n_cols_per_pred
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

    def run(self,
            features_data: dict,
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
        pred_count = self.get_predcount()
        self.tm.compute(active_columns, learn=learn)
        # Get anomaly metrics
        anomaly_score = self.tm.anomaly
        anomaly_likelihood = self.anomaly_history.compute(anomaly_score)
        # Ensure pred_count > 0 when anomaly_score < 1.0
        if anomaly_score < 1.0 and pred_count == 0:
            raise RuntimeError(f"0 preds with anomaly={anomaly_score}")

        # PREDICTOR
        # Predict raw feature value -- IF enabled AND model is 1 feature (excluding timestamp)
        steps_predictions = {}
        features_nontimestamp = {k: v for k, v in self.features_enc_params.items()
                                 if v['type'] not in self.types_time}
        if predictor_config['enable'] and len(features_nontimestamp) == 1:
            feature = list(features_nontimestamp.keys())[0]
            steps_predictions = self.get_preds(timestep=timestep,
                                               f_data=features_data[feature])

        return anomaly_score, anomaly_likelihood, pred_count, steps_predictions
