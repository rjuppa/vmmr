#!/usr/bin/env python3

# create a sample dataset
# with 5 car producers
# use just 1000 images per producer

import os
import shutil
from unidecode import unidecode
from os import listdir
from os.path import isfile, join, isdir


root = "/Users/radekj/devroot/vmmr/datasets/dest"
sample5 = "/Users/radekj/devroot/vmmr/datasets/sample52"

producers = ["Audi", "Ford", "Opel", "Skoda", "Volkswagen"]


def copy_files(producer, path, count):
    model_names = os.listdir(path)
    i = 0
    for model in model_names:
        model_path = join(path, model)
        if os.path.isdir(model_path):
            dir_dest = join(sample5, producer)
            if not os.path.exists(dir_dest):
                os.mkdir(dir_dest)

            for name in os.listdir(model_path):
                filename = join(model_path, name)
                file_dest = join(dir_dest, name)
                if os.path.isfile(filename) and filename[-4:] == ".jpg":
                    if not os.path.exists(file_dest):
                        shutil.copy(filename, file_dest)
                        i += 1
                if i > count:
                    return


if __name__ == "__main__":
    print("Start..")
    for producer in listdir(root):
        if producer in producers:
            sub_path = join(root, producer)
            print(producer)
            if isdir(sub_path):
                models = os.listdir(root)
                copy_files(producer, sub_path, 2000)
