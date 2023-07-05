from __future__ import annotations
import argparse
import itertools
import random

from collections.abc import Iterator, Mapping
from typing import TypeVar, overload, Any, List, Iterable, Sequence, MutableSequence

import numpy as np

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

    dicts = list(filter(lambda x: x is not None, dicts))

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


def choose_features(features: Sequence[Any], n_choices: int, choice_size: int) -> List[tuple]:
    all_comb = list(itertools.combinations(features, choice_size))

    if n_choices < 1:
        return []

    if n_choices > len(all_comb):
        raise ArithmeticError(f"'n_choices' impossibly big, maximum is {len(all_comb)}")

    elif n_choices == len(all_comb):
        return all_comb

    else:
        chosen_indices = np.random.choice(len(all_comb), size=n_choices, replace=False)
        return [all_comb[i] for i in chosen_indices]


def split_features(features: MutableSequence[Any], size: int, shuffle: bool = False) -> List[tuple]:
    if shuffle:
        random.shuffle(features)

    return [tuple(features[i:i + size]) for i in range(0, len(features), size)]


def get_base_names(feature: str, fjs: str) -> list[str]:
    return feature.split(fjs)


def get_joined_name(features: list[str] | tuple[str], fjs: str) -> str:
    return fjs.join(features)
