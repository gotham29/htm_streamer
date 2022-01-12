import argparse
import json
import os
import pandas as pd
import pickle


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


def make_dir(mydir):
    """
    Purpose:
        Make directory
    Inputs:
        mydir
            type: sty
            meaning: path to dir to make
    Outputs:
        none (dir made)
    """
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def load_json(path_json):
    """
    Purpose:
        Load json file
    Inputs:
        path_json
            type: str
            meaning: path to load json file from
    Outputs:
        data
            type: dict
            meaning: json data loaded
    """
    f = open(path_json, "r")
    data = json.loads(f.read())
    f.close()
    return data


def save_json(data, path_json):
    """
    Purpose:
        Save data to json file
    Inputs:
        data
            type: dict
            meaning: data to save in json format
        path_json
            type: str
            meaning: path to json file
    Outputs:
        none (json written)
    """
    json_object = json.dumps(data, indent=4)
    with open(path_json, "w") as outfile:
        outfile.write(json_object)


def save_data_as_pickle(data_struct, f_path):
    """
    Purpose:
        Save data to pkl file
    Inputs:
        data_struct
            type: any
            meaning: data to save in pkl format
        f_path
            type: str
            meaning: path to pkl file
    Outputs:
        True
            type: bool
            meaning: function ran
    """
    with open(f_path, 'wb') as handle:
        pickle.dump(data_struct, handle)
    return True


def load_pickle_object_as_data(file_path):
    """
    Purpose:
        Load data from pkl file
    Inputs:
        file_path
            type: str
            meaning: path to pkl file
    Outputs:
        data
            type: pkl
            meaning: pkl data loaded
    """
    with open(file_path, 'rb') as f_handle:
        data = pickle.load(f_handle)
    return data


def save_models(features_models, dir_models):
    """
    Purpose:
        Save models to pkl
    Inputs:
        features_models
            type: dict
            meaning: model obj for each feature
        dir_models
            type: str
            meaning: path to dir where pkl models are written to
    Outputs:
        none (pkls written)
    """
    for t, model in features_models.items():
        path_model = os.path.join(dir_models, f"{t}.pkl")
        save_data_as_pickle(model, path_model)


def load_models(dir_models):
    """
    Purpose:
        Load pkl models for each feature from dir
    Inputs:
        dir_models
            type: str
            meaning: path to dir where pkl models are loaded from
    Outputs:
        features_models
            type: dict
            meaning: model obj for each feature
    """
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    features_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        features_models[f.replace('.pkl', '')] = model
    return features_models


def save_outputs(features_outputs, timestep_current, timestep_sampling, dir_out):
    """
    Purpose:
        Save model outputs for all features (json)
    Inputs:
        timestep
            type: int
            meaning: current timestep
        features_outputs
            type: dict
            meaning: model outputs for each feature
        dir_out
            type: str
            meaning: path to dir where json outputs are written to
    Outputs:
        none (jsons written)
    """
    for f, output in features_outputs.items():
        result_current = pd.DataFrame({k: [v] for k, v in output.items()})
        path_result_total = os.path.join(dir_out, f"{f}.csv")

        first_output = True if (timestep_current == 1+timestep_sampling) else False
        if first_output:
            result_total = result_current
        else:
            result_total = pd.concat([pd.read_csv(path_result_total), result_current], axis=0)

        result_total.to_csv(path_result_total, index=False)
