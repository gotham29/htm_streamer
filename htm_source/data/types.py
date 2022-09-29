from enum import Enum


# DEFINE ALLOWED TYPES HERE
__ENCODABLE_NUMERIC = frozenset(['int', 'float'])
__ENCODABLE_DATETIME = frozenset(['time', 'date', 'datetime', 'timestamp'])


class HTMType(Enum):
    Numeric = 0
    Datetime = 1


def to_htm_type(dtype: str, /) -> HTMType:
    """
    Converts one of the allowed str types to an HTMType
    """
    if dtype in __ENCODABLE_NUMERIC:
        return HTMType.Numeric
    elif dtype in __ENCODABLE_DATETIME:
        return HTMType.Datetime

    # future implementations here..

    else:
        raise TypeError(f"Type '{dtype}' has no equivalent HTMType.")
