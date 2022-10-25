from __future__ import annotations
import argparse

from collections.abc import Iterator, Mapping
from typing import TypeVar, overload, Any


"""
This is just some typing stuff for dict_zip() func.
"""

_K = TypeVar('_K')

_V1 = TypeVar('_V1')
_V2 = TypeVar('_V2')
_V3 = TypeVar('_V3')


@overload
def dict_zip(
        m1: Mapping[_K, _V1],
        m2: Mapping[_K, _V2],
) -> Iterator[tuple[_K, _V1, _V2]]:
    ...


@overload
def dict_zip(
        m1: Mapping[_K, _V1],
        m2: Mapping[_K, _V2],
        m3: Mapping[_K, _V3],
) -> Iterator[tuple[_K, _V1, _V2, _V3]]:
    ...


def dict_zip(*dicts):
    """
    Will zip any number of dictionaries with shared keys, yielding each time:
    key, d1_val, d2_val, ...

    Fails if not all dictionaries have exactly the same keys
    """
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError('arguments must have the same length')

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def check_for_missing_features(row, features_expected: list) -> list:
    features_missing = [f for f in features_expected if f not in row]
    return features_missing


def get_args():
    """
    Purpose:
        Load module args
    Inputs:
        none
    Outputs:
        args
            type: dict
            meaning: object containing module arg values
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config_path_user', required=True,
                        help='path to yaml config')

    parser.add_argument('-cd', '--config_path_model', required=True,
                        help='path to yaml default config')

    parser.add_argument('-d', '--data_path', required=True,
                        help='path to json data file')

    parser.add_argument('-ds', '--data_stream_dir', required=False,
                        help='dir to save stream data to')

    parser.add_argument('-m', '--models_dir', required=True,
                        help='dir to store models')

    parser.add_argument('-o', '--outputs_dir', required=True,
                        help='dir to store outputs')

    return parser.parse_args()


def isnumeric(val: Any) -> bool:
    return isinstance(val, (int, float))
