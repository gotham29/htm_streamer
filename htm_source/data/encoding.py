import os
import sys

from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE_Parameters, RDSE

from htm_source.data.types import HTMType

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from logger import get_logger

log = get_logger(__name__)


def init_rdse(rdse_params, max_fail=3):
    encoder = None
    counter = 0
    while encoder is None:
        try:
            encoder = RDSE(rdse_params)
        except RuntimeError as e:
            counter += 1
            if counter == max_fail:
                log.error(
                    msg=f"Failed RDSE random collision check {max_fail} times\n  change rdse params --> {rdse_params}")
                raise e
            pass
    return encoder


class EncoderFactory:
    @staticmethod
    def get_encoder(dtype: HTMType, encoder_params: dict):
        """
        Returns the appropriate encoder based on given HTMType and parameters dict
        """
        # if dtype is HTMType.Numeric:
        if dtype in [HTMType.Numeric, HTMType.Categoric]:
            rdse_params = RDSE_Parameters()
            rdse_params.seed = encoder_params['seed']
            rdse_params.size = encoder_params['size']
            rdse_params.activeBits = encoder_params["activeBits"]
            if dtype is HTMType.Numeric:
                rdse_params.resolution = encoder_params['resolution']
            else:  # dtype is HTMType.Categoric
                rdse_params.category = True
            encoder = init_rdse(rdse_params)

        elif dtype is HTMType.Datetime:
            encoder = DateEncoder(timeOfDay=encoder_params["timeOfDay"],
                                  weekend=encoder_params["weekend"],
                                  dayOfWeek=encoder_params["dayOfWeek"],
                                  holiday=encoder_params["holiday"],
                                  season=encoder_params["season"])

        # future implementations here..

        else:
            raise NotImplementedError(f"Encoder not implemented for '{dtype}'")

        return encoder
