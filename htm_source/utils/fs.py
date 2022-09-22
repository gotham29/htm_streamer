import json
import os
import pickle

import pandas as pd
import yaml


def load_json(path_json: str) -> dict:
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


def save_json(data, path_json: str):
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


def save_data_as_pickle(data_struct, f_path: str):
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


def load_pickle_object_as_data(file_path: str):
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


def save_models(features_models: dict, dir_models: str):
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


def load_models(dir_models: str):
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


def save_outputs(features_outputs: dict, timestep_init: int, timestep_current: int, save_outputs_accumulated: bool,
                 dir_out: str):
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

    # first_output = False if (timestep_current > 1 + timestep_sampling) else True
    first_output = True if (timestep_current == 1 + timestep_init) else False
    for f, output in features_outputs.items():

        # Save current output
        result_current = pd.DataFrame({k: [v] for k, v in output.items()})
        path_result_current = os.path.join(dir_out, f"{f}--timestep={timestep_current}.csv")
        result_current.to_csv(path_result_current)
        # Delete prior output
        path_result_previous = os.path.join(dir_out, f"{f}--timestep={timestep_current - 1}.csv")
        if os.path.exists(path_result_previous):
            os.remove(path_result_previous)
        result_current.to_csv(path_result_current)

        # Save total output
        if save_outputs_accumulated:
            result_total = result_current
            path_result_total = os.path.join(dir_out, f"{f}.csv")
            if not first_output:
                result_total = pd.concat([pd.read_csv(path_result_total), result_current], axis=0)
            result_total.to_csv(path_result_total, index=False)
    if first_output:
        print(f"  Save Outputs Accumulated = {save_outputs_accumulated}\n")


def load_config(yaml_path: str) -> dict:
    """
    Purpose:
        Load config from path
    Inputs:
        yaml_path
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_config(cfg: dict, yaml_path: str) -> dict:
    """
    Purpose:
        Save config to path
    Inputs:
        cfg
            type: dict
            meaning: config (yaml)
        yaml_path
            type: str
            meaning: .yaml path to save to
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- saved
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    return cfg
