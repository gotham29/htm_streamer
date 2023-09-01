from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from htm_source.data import AnomalyLikelihood
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.sdr import SDR

from htm_source.utils.sdr import sdr_max_pool, flatten_shape, squeeze_shape


class HTMBase(ABC):
    def __init__(self, *args, **kwargs):
        self._iteration = 0
        self._seed = None
        self._out_dims = None
        self._learning = True
        self._configuration = dict()
        self._anomaly_history = {'score': [], 'likelihood': []}
        self._initialized_sp = False
        self.lr_sch = None
        self.sp = None
        self.tm = None
        self.al = None
        self.max_pool = None
        self.flat = None

    def __call__(self, *args, **kwargs) -> SDR:
        # pre-forward hooks
        self.learning_hook()

        ret_val = self.forward(*args, **kwargs)

        # post-forward hooks
        ret_val = self.post_forward(ret_val)
        self._iteration += 1
        return ret_val

    def _init_sp(self):
        raise NotImplementedError

    def _init_tm(self):
        raise NotImplementedError

    def post_forward(self, *args):
        return args

    def forward(self, *args, **kwargs) -> SDR:
        raise NotImplementedError

    def train(self):
        self._learning = True

    def eval(self):
        self._learning = False

    def learning_hook(self):
        if self.lr_sch is None:
            return
        if self._iteration in self.lr_sch:
            self.train()
        else:
            self.eval()

    @property
    def learning(self) -> bool:
        return self._learning

    @property
    def config(self) -> dict:
        return self._configuration.copy()

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def timestep(self) -> int:
        return self._iteration

    @property
    def output_dim(self) -> np.ndarray:
        ret_val = self._out_dims
        if self.max_pool not in (None, False, 1, 0):
            ret_val = tuple((ret_val[0] // self.max_pool, *ret_val[1:]))
        if self.flat:
            ret_val = squeeze_shape(ret_val)
        return ret_val

    @property
    def anomaly(self) -> dict:
        return self._anomaly_history.copy()


class HTMModule(HTMBase):
    def __init__(self, input_dims: tuple | list | np.ndarray,
                 sp_cfg: dict | None,
                 tm_cfg: dict,
                 seed: int = 0,
                 max_pool: int = 1,
                 flatten: bool = False,
                 learn_schedule: slice = None,
                 anomaly_score: bool = True,
                 anomaly_likelihood: bool = False,
                 al_learn_period: int = 1000,
                 lazy_init: bool = True):

        super().__init__()
        self._configuration = {'sp': sp_cfg, 'tm': tm_cfg}
        self._seed = seed
        self._input_dims = input_dims
        if not lazy_init:
            self._init_sp()

        self._init_tm()
        self._out_dims = tuple((*self._get_column_dims(), self.config['tm']['cellsPerColumn']))
        self.predictive_cells = SDR(dimensions=self._out_dims)
        self.max_pool = max_pool
        self.flat = flatten
        self.calc_anomaly = anomaly_score
        self.lr_sch = learn_schedule
        self.al_learn_period = al_learn_period

        if anomaly_likelihood:
            self._init_al()

    def _init_sp(self):
        if self.config['sp'] is not None:
            self.sp = SpatialPooler(inputDimensions=list(self._input_dims),
                                    columnDimensions=self.config["sp"]["columnDimensions"],
                                    potentialPct=self.config["sp"]["potentialPct"],
                                    potentialRadius=self.config["sp"]["potentialRadius"],
                                    globalInhibition=self.config["sp"]["globalInhibition"],
                                    synPermInactiveDec=self.config["sp"]["synPermInactiveDec"],
                                    stimulusThreshold=self.config["sp"]["stimulusThreshold"],
                                    synPermActiveInc=self.config["sp"]["synPermActiveInc"],
                                    synPermConnected=self.config["sp"]["synPermConnected"],
                                    boostStrength=self.config["sp"]["boostStrength"],
                                    localAreaDensity=self.config['sp']['localAreaDensity'],
                                    wrapAround=self.config['sp']['wrapAround'],
                                    seed=self.seed)

    def _init_tm(self):
        column_dimensions = self._get_column_dims()
        self.tm = TemporalMemory(columnDimensions=column_dimensions,
                                 cellsPerColumn=self.config["tm"]["cellsPerColumn"],
                                 activationThreshold=self.config["tm"]["activationThreshold"],
                                 initialPermanence=self.config["tm"]["initialPerm"],
                                 connectedPermanence=self.config["tm"]["permanenceConnected"],
                                 minThreshold=self.config["tm"]["minThreshold"],
                                 maxNewSynapseCount=self.config["tm"]["newSynapseCount"],
                                 permanenceIncrement=self.config["tm"]["permanenceInc"],
                                 permanenceDecrement=self.config["tm"]["permanenceDec"],
                                 predictedSegmentDecrement=self.config["tm"]["predictedSegmentDecrement"],
                                 maxSegmentsPerCell=self.config["tm"]["maxSegmentsPerCell"],
                                 maxSynapsesPerSegment=self.config["tm"]["maxSynapsesPerSegment"],
                                 seed=self.seed)

    def _init_al(self):
        self.al = AnomalyLikelihood(learningPeriod=max(self.al_learn_period - 100, 0))

    def _get_column_dims(self) -> np.ndarray:
        return np.array(self.config["sp"]["columnDimensions"] if self.config['sp'] else list(self._input_dims))

    def make_prediction(self):
        self.tm.activateDendrites(learn=False)
        self.predictive_cells = self.tm.getPredictiveCells()

    def forward(self, input_sdr: SDR, input_value: Any = None) -> SDR:

        # SPATIAL POOLER (or just encoding)
        if self.sp:
            # Create an SDR to represent active columns
            active_columns = SDR(self.sp.getColumnDimensions())
            self.sp.compute(input_sdr, self.learning, active_columns)
        else:
            active_columns = input_sdr

        # TEMPORAL MEMORY
        self.tm.activateDendrites(learn=self.learning)
        self.tm.activateCells(activeColumns=active_columns, learn=self.learning)
        winner_cells: SDR = self.tm.getWinnerCells()

        # check anomaly
        if self.calc_anomaly:
            anomaly_score = 1. - self.predictive_cells.getOverlap(winner_cells) / (winner_cells.getSum() + 1e-8)
            self._anomaly_history['score'].append(anomaly_score)
            if self.al:
                if input_value is None:
                    raise ValueError(f"`input_value` must be specified if wanting to calculate anomaly likelihood")

                self._anomaly_history['likelihood'].append(self.al.anomalyProbability(value=input_value,
                                                                                      anomalyScore=anomaly_score,
                                                                                      timestamp=self.timestep))

        # predict next input
        self.make_prediction()

        return self.tm.getActiveCells()

    def post_forward(self, x: SDR) -> SDR:
        x = sdr_max_pool(x, ratio=self.max_pool, axis=-1)
        if self.flat:
            new_shape = squeeze_shape(x.dimensions)
            x.reshape(new_shape)
        return x

    def summary(self) -> str:
        return f"[{self._input_dims} --> {self.output_dim}] (max_pool: {self.max_pool})"
