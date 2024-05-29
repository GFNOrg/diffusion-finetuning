import shutil
import numpy as np
import os
import platform
import torch
import json
from pathlib import Path
from os import path as pt
from pprint import pprint
from collections.abc import Iterable
import zipfile


def file_exist_query(filename):
    path = Path(filename)
    if path.is_file():
        res = None
        while res not in ['y', 'Y', 'n', 'N']:
            res = input("\nThe file in '{}' already exists, do you really wish to re-write its contents? [y/n]".format(filename))
            if res not in ['y', 'Y', 'n', 'N']:
                print("Please reply with 'y' or 'n'")
        if res in ['n', 'N']:
            return False
    return True


def file_exists(filename):
    path = Path(filename)
    if path.is_file():
        return True
    return False


def remove_file(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False


def remove_files(folder, file_ending='', file_beginning='', contains=''):
    files = get_filenames(folder, starts_with=file_beginning, contains=contains, ends_with=file_ending)
    while len(files) > 0:
        try:
            for file in files:
                remove_file(folder + file)
            files = get_filenames(folder, starts_with=file_beginning, contains=contains, ends_with=file_ending)
        except Exception as e:
            print(e)
            print(f'the file {e} could not be removed, proceeding as normal...')


def folder_exists(folder_name):
    return pt.isdir(folder_name)


def folder_create(folder_name, exist_ok=False, parents=True):
    path = Path(folder_name)
    try:
        if exist_ok and folder_exists(folder_name):
            return True
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise e
    return True


def remove_directory(path):
    if folder_exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
        return True
    else:
        return False


# get all filenames in a folder with particular attributes
def get_filenames(path="./", ends_with='', starts_with='', contains=[], excludes=[], ignore_capitals=True, recursive=False, get_associated_path=False):

    if isinstance(contains, str):
        contains = [contains]

    if isinstance(excludes, str):
        excludes = [excludes]

    if not recursive:
        if ignore_capitals:
            return list(np.sort([file for file in os.listdir(path)
                                 if file.lower().endswith(ends_with.lower())
                                 and file.lower().startswith(starts_with.lower())
                                 and all([con.lower() in file.lower() for con in contains])
                                 and all([exc.lower() not in file.lower() for exc in excludes])]))
        else:
            return list(np.sort([file for file in os.listdir(path)
                                 if file.endswith(ends_with)
                                 and file.startswith(starts_with)
                                 and all([con in file for con in contains])
                                 and all([exc not in file for exc in excludes])]))
    else:
        files = []
        paths = []
        for (root, _, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith(ends_with) \
                        and filename.startswith(starts_with) \
                        and any([con in filename for con in contains]):
                    files.append(filename)
                    if get_associated_path:
                        paths.append(f"{root}/{filename}")

        if not get_associated_path:
            return files
        else:
            return paths, files


# retrieves date of creating of a file (path) passed in input
def creation_date(path_to_file):
    """
    # retrieves date of creating of a file (path) passed in input
    :param path_to_file:
    :return: date created
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            return stat.st_mtime


def ask_user_model_load(save_folder="", model_name=""):
    train = False
    paths, existing_model_names = get_filenames(
        save_folder,
        recursive=True,
        ends_with='.pth',
        get_associated_path=True
    )
    matching_models = [existing_name for existing_name in existing_model_names if model_name in existing_name]
    while True:
        if len(matching_models) > 0:
            print("\nThe following models were found:\n")
            for i, (model_path, model_name) in enumerate(zip(paths, existing_model_names)):
                print(f"    {i}. {model_name}")

            usr_in = 'n'
            while not usr_in[0].isdigit() and usr_in[0].lower() != 'q':
                usr_in = input("\nselect model number to load, or 'q' to re-train: ")

            if usr_in[0].lower() == 'q':
                train = True
                break
            else:
                if int(usr_in) < len(paths):
                    print('loading model...', end='')
                    net = torch.load(f"{paths[int(usr_in)]}")
                    print('loaded')
                    return train, net
                else:
                    print("\nThe value inserted is not in the list, please try again.")
        else:
            train = True
            break

    return train, None


def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to a specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # Remove the zip file after extraction


def save_dataframe_dict(dataset, dataset_name, path=''):
    # dataset = copy.deepcopy(dataset)
    #
    # dataset = {key: data.to_list() for key, data in dataset.items()}
    # pd.DataFrame.from_dict(dataset, orient='index')
    # dataset.apply(lambda x: x[''], column=1)
    try:
        print(f"saving dataframe dictionary for '{dataset_name}'")
        np.save(f"{path}ppm", dataset)
        print("done")
    except Exception as e:
        raise Exception
    return True


def load_dataframe_dict(dataset_name, path=''):
    print(f"loading dataframe dictionary for '{dataset_name}'")
    filenames = get_filenames(path, ends_with='ppm.npy')
    if len(filenames) > 0:
        data = np.load(f"{path}{filenames[0]}", allow_pickle=True)
        print("done")
        return data.item()
    else:
        return None


def save_dict_to_file(data=None, path=None, filename=None, format='json', replace=True):
    # if don't want to overwrite, check next available name
    file = "{}{}.{}".format(path, filename, format)
    if not replace:
        num = 0
        while file_exists(file):
            file = "{}{}-({}).{}".format(path, filename, num, format)
            num += 1
    try:
        with open(file, 'w') as exp_file:
            json.dump(data, exp_file)
    except Exception as e:
        print(f"WARNING: could not save FILE {filename} at {path}")
        print(f"{e.__class__.__name__}: {e}")


def load_dict_from_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def results_printer(results, filename, folder, experiment_tag="", header="", show=True):

    def save_write_file(file_str):
        if file_exists(file_str):
            file = open(f"{file_str}", "a+")
        else:
            file = open(f"{file_str}", "w")
        file.write(f" ***** experiment: {experiment_tag} ***** \n\n")
        file.write(header + "\n")
        pprint(results, file)
        file.write("\n\n\n")
        file.close()

        if show:
            print(header)
            pprint(results)

    if isinstance(folder, list):
        for fld in folder:
            file_str = f"{fld}{filename}.txt"
            save_write_file(file_str)
    else:
        file_str = f"{folder}{filename}.txt"
        save_write_file(file_str)


class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)