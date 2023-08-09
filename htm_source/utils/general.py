from __future__ import annotations
import argparse
import itertools
import random
from collections.abc import Iterator, Mapping
from typing import TypeVar, overload, Any, List, Sequence, MutableSequence, Tuple

import numba as nb
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
) -> Iterator[Tuple[_K, _V1, _V2]]:
    ...


@overload
def dict_zip(
        m1: Mapping[_K, _V1],
        m2: Mapping[_K, _V2],
        m3: Mapping[_K, _V3],
) -> Iterator[Tuple[_K, _V1, _V2, _V3]]:
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


def choose_features(features: Sequence[str], n_choices: int, choice_size: int) -> List[Tuple[str]]:
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


def split_features(features: MutableSequence[str], size: int, shuffle: bool = False) -> List[Tuple[str]]:
    if shuffle:
        random.shuffle(features)

    return [tuple(features[i:i + size]) for i in range(0, len(features), size)]


def get_base_names(feature: str, fjs: str) -> list[str]:
    return feature.split(fjs)


def get_joined_name(features: list[str] | tuple[str], fjs: str) -> str:
    return fjs.join(features)


@nb.jit(nopython=True, nogil=True, cache=True)
def find_first(vec: np.ndarray, value) -> int:
    """ Given a vector, find and return index of first occurrence of `value` """
    for i in range(len(vec)):
        if vec[i] == value:
            return i
    return -1


@nb.jit(nopython=True, nogil=True, cache=True)
def make_windows(array: np.ndarray):
    """ Fast numba implementation.
        Given a binary array, extracts the slices of continuous 1s.
        i.e: array = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
        out: {(2, 5), (6, 7), (9, 11)} """
    current_end_index = 0
    done = False
    windows = set()

    while not done:
        # open window
        start = find_first(array[current_end_index:], 1)
        if start == -1:
            break
        else:
            start = start + current_end_index
            current_start_index = start

        # close window
        end = find_first(array[current_start_index:], 0)
        if end == -1:
            end = len(array)
            done = True
        else:
            end = end + current_start_index
            current_end_index = end

        windows.add((start, end))

    return windows
