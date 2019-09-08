#!/usr/bin/env python3

# split dataset to: test, train, valid

import os
import math
import shutil
from os import listdir
from os.path import isfile, join, isdir


parent_path = os.path.abspath('.')
src = "/storage/plzen1/home/radekj/vmmr/datasets/sample61/"

TRAIN = 0.6
TEST = 0.2
VALID = 0.2


def copy_file(files, class_name, folder):
    path = join(src, folder)
    if not os.path.exists(path):
        os.mkdir(path)

    path = join(path, class_name)
    if not os.path.exists(path):
        os.mkdir(path)

    for filepath in files:
        name = filepath.split("/")[-1]
        dest_path = join(path, name)
        shutil.copy(filepath, dest_path)


def load_class_folder(class_path):
    files = []
    for name in os.listdir(class_path):
        filename = join(class_path, name)
        files.append(filename)

    return files


if __name__ == "__main__":
    print("Start..")
    data_path = join(src, "data")
    for class_name in sorted(listdir(data_path)):
        print("\n")
        print(class_name)
        class_path = join(data_path, class_name)
        if isdir(class_path):

            files = load_class_folder(class_path)
            c0 = math.floor(len(files) * VALID)
            print("{}: {}".format("valid", c0))
            copy_file(files[:c0], class_name, "valid")

            c1 = math.floor(len(files) * TEST)
            print("{}: {}".format("test", c1))
            copy_file(files[c0+1:c0+c1], class_name, "test")

            print("{}: {}".format("train", len(files) - (c0 + c1)))
            copy_file(files[c0+c1+1:], class_name, "train")
