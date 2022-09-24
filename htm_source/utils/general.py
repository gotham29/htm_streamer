import argparse

import numpy as np


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

    parser.add_argument('-c', '--config_path', required=True,
                        help='path to yaml config')

    parser.add_argument('-d', '--data_path', required=True,
                        help='path to json data file')

    parser.add_argument('-m', '--models_dir', required=True,
                        help='dir to store models')

    parser.add_argument('-o', '--outputs_dir', required=True,
                        help='dir to store outputs')

    return parser.parse_args()


def isnumeric(val):
    return isinstance(val, (int, float))
