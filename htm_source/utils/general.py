import argparse

import numpy as np


# TODO: I don't think its a good name, function implicitly checks if float.. we can probably rename or rewrite it
def check_for_missing_features(row, features_expected: list):
    features_missing = [f for f in features_expected if f not in row]
    return features_missing


# def check_for_missing_features(row, features_expected: list):
#     # check which features are numeric
#     features_numeric = {}
#     for f, v in dict(row).items():
#         try:
#             f_float = float(v)
#             if not np.isnan(f_float):
#                 features_numeric[f] = f_float
#         except:
#             pass
#
#     # check if any expected features aren't valid numeric
#     features_missing = [f for f in features_expected if f not in features_numeric]
#     return features_missing


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
