#!/usr/bin/env python3

# check empty directories in cleaned data
# and check appropriate dir. in source folders
# and copy all unprocessed data out to special folder for processing

import os
import shutil
from os import listdir
from os.path import join, isdir


parent_path = os.path.abspath('.')

# /storage/plzen1/home/radekj/vmmr/datasets
path_data = "/storage/plzen1/home/radekj/vmmr/datasets"
path_dest = "{}/dest".format(path_data)
path_src = "{}/source".format(path_data)
path_fix = "{}/fix1".format(path_data)


def get_dir_name(name):
    new_name = str(name)
    new_name = new_name.replace(" ", "_")
    new_name = new_name.replace("__", "_")
    new_name = new_name.replace("__", "_")
    return new_name


def copy_folder(make, model):
    d_path = os.path.join(path_fix, make)
    if not os.path.exists(d_path):
        os.mkdir(d_path)

    d_path = os.path.join(d_path, model)
    d_path = get_dir_name(d_path)
    if not os.path.exists(d_path):
        os.mkdir(d_path)

    dirs = ["images_cz", "images_sk", "images_pl", "images_hu", "images_cz2", "aaa_auto1/images", "aaa_auto2/images"]
    for src_name in dirs:
        s_path = "{}/{}".format(path_src, src_name)
        s_path = os.path.join(s_path, make)
        if not os.path.exists(s_path):
            continue

        s_path = os.path.join(s_path, model)
        if not os.path.exists(s_path):
            continue

        files = os.listdir(s_path)
        for file in files:
            file_path = os.path.join(s_path, file)
            if os.path.isfile(file_path) and file_path[-4:] == ".jpg":
                shutil.copy(file_path, d_path)


def get_model_images_count(path):
    image_count = 0
    for name in os.listdir(path):
        sub_path = join(path, name)
        if os.path.isfile(sub_path) and sub_path[-4:] == ".jpg":
            image_count += 1
            # print(sub_path)     # TODO print dataset

    return image_count


def get_make_count(path):
    model_count = 0
    table = {}
    for name in os.listdir(path):
        sub_path = join(path, name)
        if os.path.isdir(sub_path):
            model_count += 1
            table[name] = get_model_images_count(sub_path)

    return model_count, table


if __name__ == "__main__":
    print("Car Make, Count of models, Count of images, Median images/model")
    print("-----------------------")

    with open("missing.txt") as f:
        content = f.readlines()

    for line in content:
        make, model, _ = line.split(",")
        print("{} - {}".format(make, model))
        copy_folder(make.strip(), model.strip())
