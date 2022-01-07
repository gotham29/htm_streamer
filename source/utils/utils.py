import argparse
import json
import os
import pickle


def get_args():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config_path', required=True,
                        help='path to yaml config')

    parser.add_argument('-d', '--data_path', required=False,
                        help='path to json data file')

    return parser.parse_args()


def make_dir(mydir):
    """
    Make directory
    Inputs:
        mydir: (str) path to dir to make
    Outputs:
        NA (dir made)
    """
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def load_json(path_json):
    f = open(path_json, "r")
    return json.loads(f.read())


def save_json(data, path_json):
    json_object = json.dumps(data, indent=4)
    with open(path_json, "w") as outfile:
        outfile.write(json_object)


def save_data_as_pickle(data_struct, f_path):
    """
    Saves a dictionary as pickle object to f_path
    :param data_struct: Data object (dict/list) you want to save
    :param f_path: File path
    :return: True flag
    """
    with open(f_path, 'wb') as handle:
        pickle.dump(data_struct, handle)
    return True


def load_pickle_object_as_data(file_path):
    """
    Loads a pickle object from path
    :param file_path: file path to load pickle object from
    :return: Returns object
    """
    with open(file_path, 'rb') as f_handle:
        data = pickle.load(f_handle)
    return data


def save_models(features_models, dir_models):
    for t, model in features_models.items():
        path_model = os.path.join(dir_models, f"{t}.pkl")
        save_data_as_pickle(model, path_model)


def load_models(dir_models):
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    features_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        features_models[f.replace('.pkl', '')] = model
    return features_models


def save_outputs(timestep, features_outputs, dir_out):
    for f, output in features_outputs.items():
        dir_out_f = os.path.join(dir_out, f)
        make_dir(dir_out_f)
        path_json = os.path.join(dir_out_f, f'iter={timestep}.json')
        save_json(output, path_json)
