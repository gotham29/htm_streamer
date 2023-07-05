import os
import sys

from htm.bindings.sdr import SDR
from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE_Parameters, RDSE

from htm_source.data.types import HTMType

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


def init_rdse(rdse_params, max_fail=5, shape=None):
    encoder = None
    counter = 0
    if shape is None:
        shape = [rdse_params.size]

    while encoder is None:
        try:
            encoder = ShapedRDSE(shape, rdse_params)
        except RuntimeError as e:
            counter += 1
            if counter == max_fail:
                log.error(
                    msg=f"Failed RDSE random collision check {max_fail} times\n  change rdse params --> {rdse_params}")
                raise RuntimeError(e)
            pass
    return encoder

# TODO identity?
class DummyEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, data):
        return data


class EncoderFactory:
    @staticmethod
    def get_encoder(dtype: HTMType, encoder_params: dict):
        """
        Returns the appropriate encoder based on given HTMType and parameters dict
        """
        if dtype in [HTMType.Numeric, HTMType.Categorical]:
            rdse_params = RDSE_Parameters()
            rdse_params.seed = encoder_params['seed']
            rdse_params.size = encoder_params['size']
            rdse_params.activeBits = encoder_params["activeBits"]
            if dtype is HTMType.Numeric:
                rdse_params.resolution = encoder_params['resolution']
            else:  # dtype is HTMType.Categorical
                rdse_params.category = True
            encoder = init_rdse(rdse_params, shape=encoder_params.get('shape', None))

        elif dtype is HTMType.Datetime:
            encoder = DateEncoder(timeOfDay=encoder_params["timeOfDay"],
                                  weekend=encoder_params["weekend"],
                                  dayOfWeek=encoder_params["dayOfWeek"],
                                  holiday=encoder_params["holiday"],
                                  season=encoder_params["season"])

        elif dtype is HTMType.SDR:
            encoder = DummyEncoder()

        # future implementations here..

        else:
            raise NotImplementedError(f"Encoder not implemented for '{dtype}'")

        return encoder


class ShapedRDSE:
    """ Like RDSE, but able to return multidimensional encodings """
    def __init__(self, shape, rdse_params):
        self.encoder = RDSE(rdse_params)
        self.shape = shape

    def encode(self, *args, **kwargs) -> SDR:
        encoding: SDR = self.encoder.encode(*args, **kwargs)
        encoding.reshape(self.shape)
        return encoding

    @property
    def size(self):
        return self.encoder.size

    @property
    def dimensions(self):
        return self.shape
