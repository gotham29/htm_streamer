import collections
import math
import numbers
import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


class Serializable(object):
    """
    Serializable base class establishing
    :meth:`~nupic.serializable.Serializable.read` and
    :meth:`~nupic.serializable.Serializable.write` abstract methods,
    :meth:`.readFromFile` and :meth:`.writeToFile` concrete methods to support
    serialization with Cap'n Proto.
    """

    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def getSchema(cls):
        """
        Get Cap'n Proto schema.
        ..warning: This is an abstract method.  Per abc protocol, attempts to subclass
                   without overriding will fail.
        @returns Cap'n Proto schema
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, proto):
        """
        Create a new object initialized from Cap'n Proto obj.
        Note: This is an abstract method.  Per abc protocol, attempts to subclass
        without overriding will fail.
        :param proto: Cap'n Proto obj
        :return: Obj initialized from proto
        """
        pass

    @abstractmethod
    def write(self, proto):
        """
        Write obj instance to Cap'n Proto object
        .. warning: This is an abstract method.  Per abc protocol, attempts to
                    subclass without overriding will fail.
        :param proto: Cap'n Proto obj
        """
        pass

    @classmethod
    def readFromFile(cls, f, packed=True):
        """
        Read serialized object from file.
        :param f: input file
        :param packed: If true, will assume content is packed
        :return: first-class instance initialized from proto obj
        """
        # Get capnproto schema from instance
        schema = cls.getSchema()

        # Read from file
        if packed:
            proto = schema.read_packed(f)
        else:
            proto = schema.read(f)

        # Return first-class instance initialized from proto obj
        return cls.read(proto)

    def writeToFile(self, f, packed=True):
        """
        Write serialized object to file.
        :param f: output file
        :param packed: If true, will pack contents.
        """
        # Get capnproto schema from instance
        schema = self.getSchema()

        # Construct new message, otherwise refered to as `proto`
        proto = schema.new_message()

        # Populate message w/ `write()` instance method
        self.write(proto)

        # Finally, write to file
        if packed:
            proto.write_packed(f)
        else:
            proto.write(f)


class MovingAverage(Serializable):
    """Helper class for computing moving average and sliding window"""

    def __init__(self, windowSize, existingHistoricalValues=None):
        """
        new instance of MovingAverage, so method .next() can be used
        @param windowSize - length of sliding window
        @param existingHistoricalValues - construct the object with already
            some values in it.
        """
        if not isinstance(windowSize, numbers.Integral):
            raise TypeError("MovingAverage - windowSize must be integer type")
        if windowSize <= 0:
            raise ValueError("MovingAverage - windowSize must be >0")

        self.windowSize = windowSize
        if existingHistoricalValues is not None:
            self.slidingWindow = existingHistoricalValues[
                                 len(existingHistoricalValues) - windowSize:]
        else:
            self.slidingWindow = []
        self.total = float(sum(self.slidingWindow))

    @staticmethod
    def compute(slidingWindow, total, newVal, windowSize):
        """Routine for computing a moving average.
        @param slidingWindow a list of previous values to use in computation that
            will be modified and returned
        @param total the sum of the values in slidingWindow to be used in the
            calculation of the moving average
        @param newVal a new number compute the new windowed average
        @param windowSize how many values to use in the moving window
        @returns an updated windowed average, the modified input slidingWindow list,
            and the new total sum of the sliding window
        """
        if len(slidingWindow) == windowSize:
            total -= slidingWindow.pop(0)

        slidingWindow.append(newVal)
        total += newVal
        return float(total) / len(slidingWindow), slidingWindow, total

    def next(self, newValue):
        """Instance method wrapper around compute."""
        newAverage, self.slidingWindow, self.total = self.compute(
            self.slidingWindow, self.total, newValue, self.windowSize)
        return newAverage

    def getSlidingWindow(self):
        return self.slidingWindow

    def getCurrentAvg(self):
        """get current average"""
        return float(self.total) / len(self.slidingWindow)

    # TODO obsoleted by capnp, will be removed in future
    def __setstate__(self, state):
        """ for loading this object"""
        self.__dict__.update(state)

        if not hasattr(self, "slidingWindow"):
            self.slidingWindow = []

        if not hasattr(self, "total"):
            self.total = 0
            self.slidingWindow = sum(self.slidingWindow)

    def __eq__(self, o):
        return (isinstance(o, MovingAverage) and
                o.slidingWindow == self.slidingWindow and
                o.total == self.total and
                o.windowSize == self.windowSize)

    def __call__(self, value):
        return self.next(value)

    @classmethod
    def read(cls, proto):
        movingAverage = object.__new__(cls)
        movingAverage.windowSize = proto.windowSize
        movingAverage.slidingWindow = list(proto.slidingWindow)
        movingAverage.total = proto.total
        return movingAverage

    def write(self, proto):
        proto.windowSize = self.windowSize
        proto.slidingWindow = self.slidingWindow
        proto.total = self.total

    # @classmethod
    # def getSchema(cls):
    #     return MovingAverageProto


class AnomalyLikelihood(Serializable):
    """
    Helper class for running anomaly likelihood computation. To use it simply
    create an instance and then feed it successive anomaly scores:
    .. code-block:: python
        anomalyLikelihood = AnomalyLikelihood()
        while still_have_data:
          # Get anomaly score from model
          # Compute probability that an anomaly has ocurred
          anomalyProbability = anomalyLikelihood.anomalyProbability(
              value, anomalyScore, timestamp)
    """

    def __init__(self,
                 claLearningPeriod=None,
                 learningPeriod=288,
                 estimationSamples=100,
                 historicWindowSize=8640,
                 reestimationPeriod=100):
        """
        NOTE: Anomaly likelihood scores are reported at a flat 0.5 for
        learningPeriod + estimationSamples iterations.
        claLearningPeriod and learningPeriod are specifying the same variable,
        although claLearningPeriod is a deprecated name for it.
        :param learningPeriod: (claLearningPeriod: deprecated) - (int) the number of
          iterations required for the algorithm to learn the basic patterns in the
          dataset and for the anomaly score to 'settle down'. The default is based
          on empirical observations but in reality this could be larger for more
          complex domains. The downside if this is too large is that real anomalies
          might get ignored and not flagged.
        :param estimationSamples: (int) the number of reasonable anomaly scores
          required for the initial estimate of the Gaussian. The default of 100
          records is reasonable - we just need sufficient samples to get a decent
          estimate for the Gaussian. It's unlikely you will need to tune this since
          the Gaussian is re-estimated every 10 iterations by default.
        :param historicWindowSize: (int) size of sliding window of historical
          data points to maintain for periodic reestimation of the Gaussian. Note:
          the default of 8640 is based on a month's worth of history at 5-minute
          intervals.
        :param reestimationPeriod: (int) how often we re-estimate the Gaussian
          distribution. The ideal is to re-estimate every iteration but this is a
          performance hit. In general the system is not very sensitive to this
          number as long as it is small relative to the total number of records
          processed.
        """
        if historicWindowSize < estimationSamples:
            raise ValueError("estimationSamples exceeds historicWindowSize")

        self._iteration = 0
        self._historicalScores = collections.deque(maxlen=historicWindowSize)
        self._distribution = None

        if claLearningPeriod != None:
            log.warning(msg="claLearningPeriod is deprecated, use learningPeriod instead.")
            self._learningPeriod = claLearningPeriod
        else:
            self._learningPeriod = learningPeriod

        self._probationaryPeriod = self._learningPeriod + estimationSamples
        self._reestimationPeriod = reestimationPeriod

    def __eq__(self, o):
        # pylint: disable=W0212
        return (isinstance(o, AnomalyLikelihood) and
                self._iteration == o._iteration and
                self._historicalScores == o._historicalScores and
                self._distribution == o._distribution and
                self._probationaryPeriod == o._probationaryPeriod and
                self._learningPeriod == o._learningPeriod and
                self._reestimationPeriod == o._reestimationPeriod)
        # pylint: enable=W0212

    def __str__(self):
        return ("AnomalyLikelihood: %s %s %s %s %s %s" % (
            self._iteration,
            self._historicalScores,
            self._distribution,
            self._probationaryPeriod,
            self._learningPeriod,
            self._reestimationPeriod))

    @staticmethod
    def computeLogLikelihood(likelihood):
        """
        Compute a log scale representation of the likelihood value. Since the
        likelihood computations return low probabilities that often go into four 9's
        or five 9's, a log value is more useful for visualization, thresholding,
        etc.
        """
        # The log formula is:
        #     Math.log(1.0000000001 - likelihood) / Math.log(1.0 - 0.9999999999)
        return math.log(1.0000000001 - likelihood) / -23.02585084720009

    @staticmethod
    def _calcSkipRecords(numIngested, windowSize, learningPeriod):
        """Return the value of skipRecords for passing to estimateAnomalyLikelihoods
        If `windowSize` is very large (bigger than the amount of data) then this
        could just return `learningPeriod`. But when some values have fallen out of
        the historical sliding window of anomaly records, then we have to take those
        into account as well so we return the `learningPeriod` minus the number
        shifted out.
        :param numIngested - (int) number of data points that have been added to the
          sliding window of historical data points.
        :param windowSize - (int) size of sliding window of historical data points.
        :param learningPeriod - (int) the number of iterations required for the
          algorithm to learn the basic patterns in the dataset and for the anomaly
          score to 'settle down'.
        """
        numShiftedOut = max(0, numIngested - windowSize)
        return min(numIngested, max(0, learningPeriod - numShiftedOut))

    # @classmethod
    # def getSchema(cls):
    #     return AnomalyLikelihoodProto

    @classmethod
    def read(cls, proto):
        """ capnp deserialization method for the anomaly likelihood object
        :param proto: (Object) capnp proto object specified in
                              nupic.regions.anomaly_likelihood.capnp
        :returns: (Object) the deserialized AnomalyLikelihood object
        """
        # pylint: disable=W0212
        anomalyLikelihood = object.__new__(cls)
        anomalyLikelihood._iteration = proto.iteration

        anomalyLikelihood._historicalScores = collections.deque(
            maxlen=proto.historicWindowSize)
        for i, score in enumerate(proto.historicalScores):
            anomalyLikelihood._historicalScores.append((i, score.value,
                                                        score.anomalyScore))
        if proto.distribution.name:  # is "" when there is no distribution.
            anomalyLikelihood._distribution = dict()
            anomalyLikelihood._distribution['distribution'] = dict()
            anomalyLikelihood._distribution['distribution']["name"] = proto.distribution.name
            anomalyLikelihood._distribution['distribution']["mean"] = proto.distribution.mean
            anomalyLikelihood._distribution['distribution']["variance"] = proto.distribution.variance
            anomalyLikelihood._distribution['distribution']["stdev"] = proto.distribution.stdev

            anomalyLikelihood._distribution["movingAverage"] = {}
            anomalyLikelihood._distribution["movingAverage"]["windowSize"] = proto.distribution.movingAverage.windowSize
            anomalyLikelihood._distribution["movingAverage"]["historicalValues"] = []
            for value in proto.distribution.movingAverage.historicalValues:
                anomalyLikelihood._distribution["movingAverage"]["historicalValues"].append(value)
            anomalyLikelihood._distribution["movingAverage"]["total"] = proto.distribution.movingAverage.total

            anomalyLikelihood._distribution["historicalLikelihoods"] = []
            for likelihood in proto.distribution.historicalLikelihoods:
                anomalyLikelihood._distribution["historicalLikelihoods"].append(likelihood)
        else:
            anomalyLikelihood._distribution = None

        anomalyLikelihood._probationaryPeriod = proto.probationaryPeriod
        anomalyLikelihood._learningPeriod = proto.learningPeriod
        anomalyLikelihood._reestimationPeriod = proto.reestimationPeriod
        # pylint: enable=W0212

        return anomalyLikelihood

    def write(self, proto):
        """ capnp serialization method for the anomaly likelihood object
        :param proto: (Object) capnp proto object specified in
                              nupic.regions.anomaly_likelihood.capnp
        """

        proto.iteration = self._iteration

        pHistScores = proto.init('historicalScores', len(self._historicalScores))
        for i, score in enumerate(list(self._historicalScores)):
            _, value, anomalyScore = score
            record = pHistScores[i]
            record.value = float(value)
            record.anomalyScore = float(anomalyScore)

        if self._distribution:
            proto.distribution.name = self._distribution["distribution"]["name"]
            proto.distribution.mean = float(self._distribution["distribution"]["mean"])
            proto.distribution.variance = float(self._distribution["distribution"]["variance"])
            proto.distribution.stdev = float(self._distribution["distribution"]["stdev"])

            proto.distribution.movingAverage.windowSize = float(self._distribution["movingAverage"]["windowSize"])

            historicalValues = self._distribution["movingAverage"]["historicalValues"]
            pHistValues = proto.distribution.movingAverage.init(
                "historicalValues", len(historicalValues))
            for i, value in enumerate(historicalValues):
                pHistValues[i] = float(value)

            # proto.distribution.movingAverage.historicalValues = self._distribution["movingAverage"]["historicalValues"]
            proto.distribution.movingAverage.total = float(self._distribution["movingAverage"]["total"])

            historicalLikelihoods = self._distribution["historicalLikelihoods"]
            pHistLikelihoods = proto.distribution.init("historicalLikelihoods",
                                                       len(historicalLikelihoods))
            for i, likelihood in enumerate(historicalLikelihoods):
                pHistLikelihoods[i] = float(likelihood)

        proto.probationaryPeriod = self._probationaryPeriod
        proto.learningPeriod = self._learningPeriod
        proto.reestimationPeriod = self._reestimationPeriod
        proto.historicWindowSize = self._historicalScores.maxlen

    def anomalyProbability(self, value, anomalyScore, timestamp=None):
        """
        Compute the probability that the current value plus anomaly score represents
        an anomaly given the historical distribution of anomaly scores. The closer
        the number is to 1, the higher the chance it is an anomaly.
        :param value: the current metric ("raw") input value, eg. "orange", or
                       '21.2' (deg. Celsius), ...
        :param anomalyScore: the current anomaly score
        :param timestamp: [optional] timestamp of the ocurrence,
                           default (None) results in using iteration step.
        :returns: the anomalyLikelihood for this record.
        """
        if timestamp is None:
            timestamp = self._iteration

        dataPoint = (timestamp, value, anomalyScore)
        # We ignore the first probationaryPeriod data points
        if self._iteration < self._probationaryPeriod:
            likelihood = 0.5
        else:
            # On a rolling basis we re-estimate the distribution
            if ((self._distribution is None) or
                    (self._iteration % self._reestimationPeriod == 0)):
                numSkipRecords = self._calcSkipRecords(
                    numIngested=self._iteration,
                    windowSize=self._historicalScores.maxlen,
                    learningPeriod=self._learningPeriod)

                _, _, self._distribution = estimateAnomalyLikelihoods(
                    self._historicalScores,
                    skipRecords=numSkipRecords)

            likelihoods, _, self._distribution = updateAnomalyLikelihoods(
                [dataPoint],
                self._distribution)

            likelihood = 1.0 - likelihoods[0]

        # Before we exit update historical scores and iteration
        self._historicalScores.append(dataPoint)
        self._iteration += 1

        return likelihood


def estimateAnomalyLikelihoods(anomalyScores,
                               averagingWindow=10,
                               skipRecords=0,
                               verbosity=0):
    """
    Given a series of anomaly scores, compute the likelihood for each score. This
    function should be called once on a bunch of historical anomaly scores for an
    initial estimate of the distribution. It should be called again every so often
    (say every 50 records) to update the estimate.
    :param anomalyScores: a list of records. Each record is a list with the
                          following three elements: [timestamp, value, score]
                          Example::
                              [datetime.datetime(2013, 8, 10, 23, 0), 6.0, 1.0]
                          For best results, the list should be between 1000
                          and 10,000 records
    :param averagingWindow: integer number of records to average over
    :param skipRecords: integer specifying number of records to skip when
                        estimating distributions. If skip records are >=
                        len(anomalyScores), a very broad distribution is returned
                        that makes everything pretty likely.
    :param verbosity: integer controlling extent of printouts for debugging
                        0 = none
                        1 = occasional information
                        2 = print every record
    :returns: 3-tuple consisting of:
              - likelihoods
                numpy array of likelihoods, one for each aggregated point
              - avgRecordList
                list of averaged input records
              - params
                a small JSON dict that contains the state of the estimator
    """
    if verbosity > 1:
        log.info(msg="In estimateAnomalyLikelihoods.")
        log.info(msg=f"Number of anomaly scores: {len(anomalyScores)}")
        log.info(msg=f"Skip records= {skipRecords}")
        log.info(msg=f"First 20: {anomalyScores[0:min(20, len(anomalyScores))]}")

    if len(anomalyScores) == 0:
        raise ValueError("Must have at least one anomalyScore")

    # Compute averaged anomaly scores
    aggRecordList, historicalValues, total = _anomalyScoreMovingAverage(
        anomalyScores,
        windowSize=averagingWindow,
        verbosity=verbosity)
    s = [r[2] for r in aggRecordList]
    dataValues = np.array(s)

    # Estimate the distribution of anomaly scores based on aggregated records
    if len(aggRecordList) <= skipRecords:
        distributionParams = nullDistribution(verbosity=verbosity)
    else:
        distributionParams = estimateNormal(dataValues[skipRecords:])

        # HACK ALERT! The HTMPredictionModel currently does not handle constant
        # metric values very well (time of day encoder changes sometimes lead to
        # unstable SDR's even though the metric is constant). Until this is
        # resolved, we explicitly detect and handle completely flat metric values by
        # reporting them as not anomalous.
        s = [r[1] for r in aggRecordList]
        # Only do this if the values are numeric
        if all([isinstance(r[1], numbers.Number) for r in aggRecordList]):
            metricValues = np.array(s)
            metricDistribution = estimateNormal(metricValues[skipRecords:],
                                                performLowerBoundCheck=False)

            if metricDistribution["variance"] < 1.5e-5:
                distributionParams = nullDistribution(verbosity=verbosity)

    # Estimate likelihoods based on this distribution
    likelihoods = np.array(dataValues, dtype=float)
    for i, s in enumerate(dataValues):
        likelihoods[i] = tailProbability(s, distributionParams)

    # Filter likelihood values
    filteredLikelihoods = np.array(
        _filterLikelihoods(likelihoods))

    params = {
        "distribution": distributionParams,
        "movingAverage": {
            "historicalValues": historicalValues,
            "total": total,
            "windowSize": averagingWindow,
        },
        "historicalLikelihoods":
            list(likelihoods[-min(averagingWindow, len(likelihoods)):]),
    }

    if verbosity > 1:
        log.info(msg=f"Discovered params=\n  {params}")
        log.info(msg=f"Number of likelihoods: {len(likelihoods)}")
        log.info(msg=f"First 20 likelihoods:\n  {filteredLikelihoods[0:min(20, len(filteredLikelihoods))]}")
        log.info(msg="leaving estimateAnomalyLikelihoods")

    return (filteredLikelihoods, aggRecordList, params)


def updateAnomalyLikelihoods(anomalyScores,
                             params,
                             verbosity=0):
    """
    Compute updated probabilities for anomalyScores using the given params.
    :param anomalyScores: a list of records. Each record is a list with the
                          following three elements: [timestamp, value, score]
                          Example::
                              [datetime.datetime(2013, 8, 10, 23, 0), 6.0, 1.0]
    :param params: the JSON dict returned by estimateAnomalyLikelihoods
    :param verbosity: integer controlling extent of printouts for debugging
    :type verbosity: int
    :returns: 3-tuple consisting of:
              - likelihoods
                numpy array of likelihoods, one for each aggregated point
              - avgRecordList
                list of averaged input records
              - params
                an updated JSON object containing the state of this metric.
    """
    if verbosity > 3:
        log.info(msg="In updateAnomalyLikelihoods.")
        log.info(msg=f"Number of anomaly scores: {len(anomalyScores)}")
        log.info(msg=f"First 20:  {anomalyScores[0:min(20, len(anomalyScores))]}")
        log.info(msg=f"Params: {params}")

    if len(anomalyScores) == 0:
        raise ValueError("Must have at least one anomalyScore")

    if not isValidEstimatorParams(params):
        raise ValueError("'params' is not a valid params structure")

    # For backward compatibility.
    if "historicalLikelihoods" not in params:
        params["historicalLikelihoods"] = [1.0]

    # Compute moving averages of these new scores using the previous values
    # as well as likelihood for these scores using the old estimator
    historicalValues = params["movingAverage"]["historicalValues"]
    total = params["movingAverage"]["total"]
    windowSize = params["movingAverage"]["windowSize"]

    aggRecordList = np.zeros(len(anomalyScores), dtype=float)
    likelihoods = np.zeros(len(anomalyScores), dtype=float)
    for i, v in enumerate(anomalyScores):
        newAverage, historicalValues, total = (
            MovingAverage.compute(historicalValues, total, v[2], windowSize)
        )
        aggRecordList[i] = newAverage
        likelihoods[i] = tailProbability(newAverage, params["distribution"])

    # Filter the likelihood values. First we prepend the historical likelihoods
    # to the current set. Then we filter the values.  We peel off the likelihoods
    # to return and the last windowSize values to store for later.
    likelihoods2 = params["historicalLikelihoods"] + list(likelihoods)
    filteredLikelihoods = _filterLikelihoods(likelihoods2)
    likelihoods[:] = filteredLikelihoods[-len(likelihoods):]
    historicalLikelihoods = likelihoods2[-min(windowSize, len(likelihoods2)):]

    # Update the estimator
    newParams = {
        "distribution": params["distribution"],
        "movingAverage": {
            "historicalValues": historicalValues,
            "total": total,
            "windowSize": windowSize,
        },
        "historicalLikelihoods": historicalLikelihoods,
    }

    if len(newParams["historicalLikelihoods"]) > windowSize:
        msg = f"historicalLikelihoods ({len(newParams['historicalLikelihoods'])}) must be >= windowSize ({windowSize})"
        log.error(msg=msg)
        raise ValueError(msg)

    if verbosity > 3:
        log.info(msg=f"Number of likelihoods: {len(likelihoods)}")
        log.info(msg=f"First 20 likelihoods: {likelihoods[0:min(20, len(likelihoods))]}")
        log.info(msg="Leaving updateAnomalyLikelihoods.")

    return (likelihoods, aggRecordList, newParams)


def _filterLikelihoods(likelihoods,
                       redThreshold=0.99999, yellowThreshold=0.999):
    """
    Filter the list of raw (pre-filtered) likelihoods so that we only preserve
    sharp increases in likelihood. 'likelihoods' can be a numpy array of floats or
    a list of floats.
    :returns: A new list of floats likelihoods containing the filtered values.
    """
    redThreshold = 1.0 - redThreshold
    yellowThreshold = 1.0 - yellowThreshold

    # The first value is untouched
    filteredLikelihoods = [likelihoods[0]]

    for i, v in enumerate(likelihoods[1:]):

        if v <= redThreshold:
            # Value is in the redzone

            if likelihoods[i] > redThreshold:
                # Previous value is not in redzone, so leave as-is
                filteredLikelihoods.append(v)
            else:
                filteredLikelihoods.append(yellowThreshold)

        else:
            # Value is below the redzone, so leave as-is
            filteredLikelihoods.append(v)

    return filteredLikelihoods


def _anomalyScoreMovingAverage(anomalyScores,
                               windowSize=10,
                               verbosity=0,
                               ):
    """
    Given a list of anomaly scores return a list of averaged records.
    anomalyScores is assumed to be a list of records of the form:
                  [datetime.datetime(2013, 8, 10, 23, 0), 6.0, 1.0]
    Each record in the returned list list contains:
        [datetime, value, averagedScore]
    *Note:* we only average the anomaly score.
    """

    historicalValues = []
    total = 0.0
    averagedRecordList = []  # Aggregated records
    for record in anomalyScores:

        # Skip (but log) records without correct number of entries
        if not isinstance(record, (list, tuple)) or len(record) != 3:
            if verbosity >= 1:
                log.info(msg=f"Mlaformed record: {record}")
            continue

        avg, historicalValues, total = (
            MovingAverage.compute(historicalValues, total, record[2], windowSize)
        )

        averagedRecordList.append([record[0], record[1], avg])

        if verbosity > 2:
            log.info(msg=f"Aggregating input record: {record}")
            log.info(msg=f"Result: {[record[0], record[1], avg]}")

    return averagedRecordList, historicalValues, total


def estimateNormal(sampleData, performLowerBoundCheck=True):
    """
    :param sampleData:
    :type sampleData: Numpy array.
    :param performLowerBoundCheck:
    :type performLowerBoundCheck: bool
    :returns: A dict containing the parameters of a normal distribution based on
        the ``sampleData``.
    """
    params = {
        "name": "normal",
        "mean": np.mean(sampleData),
        "variance": np.var(sampleData),
    }

    if performLowerBoundCheck:
        # Handle edge case of almost no deviations and super low anomaly scores. We
        # find that such low anomaly means can happen, but then the slightest blip
        # of anomaly score can cause the likelihood to jump up to red.
        if params["mean"] < 0.03:
            params["mean"] = 0.03

        # Catch all for super low variance to handle numerical precision issues
        if params["variance"] < 0.0003:
            params["variance"] = 0.0003

    # Compute standard deviation
    if params["variance"] > 0:
        params["stdev"] = math.sqrt(params["variance"])
    else:
        params["stdev"] = 0

    return params


def nullDistribution(verbosity=0):
    """
    :param verbosity: integer controlling extent of printouts for debugging
    :type verbosity: int
    :returns: A distribution that is very broad and makes every anomaly score
        between 0 and 1 pretty likely.
    """
    if verbosity > 0:
        log.info(msg="Returning nullDistribution")
    return {
        "name": "normal",
        "mean": 0.5,
        "variance": 1e6,
        "stdev": 1e3,
    }


def tailProbability(x, distributionParams):
    """
    Given the normal distribution specified by the mean and standard deviation
    in distributionParams, return the probability of getting samples further
    from the mean. For values above the mean, this is the probability of getting
    samples > x and for values below the mean, the probability of getting
    samples < x. This is the Q-function: the tail probability of the normal distribution.
    :param distributionParams: dict with 'mean' and 'stdev' of the distribution
    """
    if "mean" not in distributionParams or "stdev" not in distributionParams:
        raise RuntimeError("Insufficient parameters to specify the distribution.")

    if x < distributionParams["mean"]:
        # Gaussian is symmetrical around mean, so flip to get the tail probability
        xp = 2 * distributionParams["mean"] - x
        return tailProbability(xp, distributionParams)

    # Calculate the Q function with the complementary error function, explained
    # here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
    z = (x - distributionParams["mean"]) / distributionParams["stdev"]
    return 0.5 * math.erfc(z / 1.4142)


def isValidEstimatorParams(p):
    """
    :returns: ``True`` if ``p`` is a valid estimator params as might be returned
      by ``estimateAnomalyLikelihoods()`` or ``updateAnomalyLikelihoods``,
      ``False`` otherwise.  Just does some basic validation.
    """
    if not isinstance(p, dict):
        return False
    if "distribution" not in p:
        return False
    if "movingAverage" not in p:
        return False
    dist = p["distribution"]
    if not ("mean" in dist and "name" in dist
            and "variance" in dist and "stdev" in dist):
        return False

    return True
