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


def save_json(mydict, path_json):
    # Serializing json
    json_object = json.dumps(mydict, indent=4)
    # Writing to sample.json
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


def save_models(targets_models, dir_models):
    for t, model in targets_models.items():
        path_model = os.path.join(dir_models, f"{t}.pkl")
        save_data_as_pickle(model, path_model)


def load_models(dir_models, targets):
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    targets_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        targets_models[f.replace('.pkl','')] = model
        # print(f'    loaded model --> {f}')
    return targets_models


def save_outputs(targets_outputs, dir_out, iter_current):
    for t, output in targets_outputs.items():
        dir_out_t = os.path.join(dir_out, t)
        make_dir(dir_out_t)
        path_json = os.path.join(dir_out_t, f'iter={iter_current}.json')
        save_json(output, path_json)