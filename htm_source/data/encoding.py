from htm.encoders.date import DateEncoder
from htm.encoders.rdse import RDSE_Parameters, RDSE

from htm_source.data.types import HTMType


class EncoderFactory:
    @staticmethod
    def get_encoder(dtype: HTMType, encoder_params: dict):
        """
        Returns the appropriate encoder based on given HTMType and parameters dict
        """
        if dtype is HTMType.Numeric:
            rdse_params = RDSE_Parameters()
            rdse_params.size = encoder_params['size']
            rdse_params.activeBits = encoder_params["activeBits"]
            # rdse_params.sparsity = encoder_params["sparsity"]
            rdse_params.resolution = encoder_params['resolution']

            encoder = None
            counter = 0
            while encoder is None:
                try:
                    encoder = RDSE(rdse_params)
                except RuntimeError as e:
                    counter += 1
                    print(f"Failed {counter} times, ", e)
                    pass

            return encoder

        elif dtype is HTMType.Datetime:
            return DateEncoder(timeOfDay=encoder_params["timeOfDay"],
                               weekend=encoder_params["weekend"],
                               dayOfWeek=encoder_params["dayOfWeek"],
                               holiday=encoder_params["holiday"],
                               season=encoder_params["season"])

        # future implementations here..

        else:
            raise NotImplementedError(f"Encoder not implemented for '{dtype}'")
