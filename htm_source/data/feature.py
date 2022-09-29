from datetime import datetime
from typing import Union, Tuple, Iterable

from dateutil.parser import parse as dateutil_parse
from htm_source.data.types import HTMType, to_htm_type
from htm_source.data.encoding import EncoderFactory
from htm.bindings.sdr import SDR


class Feature:
    def __init__(self, name: str, params: dict):
        """
        This is a wrapper class for easy type checking and encoding of features
        """
        self._type = to_htm_type(params['type'])
        self._name = name
        self._params = params
        self._encoder = EncoderFactory.get_encoder(self._type, params)

        if self.type is HTMType.Datetime:
            try:
                self._dt_format = self._params['format']
            except KeyError:
                raise ValueError(f"Datetime-like feature `{self.name}` must have a `format` parameter")

    def encode(self, data: Union[str, int, float, datetime]) -> SDR:
        """
        Encodes input `data` with the appropriate encoder, based on `params` given in init
        """
        if self.type is HTMType.Datetime and not isinstance(data, datetime):
            data = datetime.strptime(data, self._dt_format)

        return self._encoder.encode(data)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __repr__(self) -> str:
        return f"Feature(name={self._name}, dtype={self._type}, params={self._params})"

    @property
    def type(self) -> HTMType:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def encoding_size(self) -> int:
        return self._encoder.size


def separate_time_and_rest(features: Iterable[Feature], strict: bool = True) -> Tuple[Union[None, str], Tuple[str, ...]]:
    """
    Given any iterable of Features, will separate the time-like feature from the rest and return the feature names:
    time_feature, (other_f_1, ...)

    If `strict` is set to True, will raise an exception if more than 1 time-like feature is found
    """
    time = None
    non_time = list()
    for feat in features:
        if feat.type is HTMType.Datetime:
            if strict and time is not None:
                raise ValueError(f"More than a single time-like feature found: {time, feat.name}")
            else:
                time = feat.name
        else:
            non_time.append(feat.name)

    return time, tuple(non_time)
