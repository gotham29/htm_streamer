from enum import Enum


# DEFINE ALLOWED TYPES HERE
__ENCODABLE_SDR = frozenset(['sdr'])
__ENCODABLE_NUMERIC = frozenset(['int', 'float'])
__ENCODABLE_DATETIME = frozenset(['time', 'date', 'datetime', 'timestamp'])
__ENCODABLE_CATEGORICAL = frozenset(['cat', 'categoric', 'categorical', 'category'])


class HTMType(Enum):
    Numeric = 0
    Datetime = 1
    Categorical = 2
    SDR = 3


def to_htm_type(dtype: str, /) -> HTMType:
    """
    Converts one of the allowed str types to an HTMType
    """
    dtype = dtype.lower()

    if dtype in __ENCODABLE_NUMERIC:
        return HTMType.Numeric
    elif dtype in __ENCODABLE_DATETIME:
        return HTMType.Datetime
    elif dtype in __ENCODABLE_CATEGORICAL:
        return HTMType.Categorical
    elif dtype in __ENCODABLE_SDR:
        return HTMType.SDR

    # future implementations here..

    else:
        raise TypeError(f"Type '{dtype}' has no equivalent HTMType.")
