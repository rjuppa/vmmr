#!/usr/bin/env python3

# create a dataset of models (merge directories)
import os
from os.path import join

import shutil

parent_path = os.path.abspath('.')
# root = "/Users/radekj/devroot/vmmr/datasets/"
root = "/storage/plzen1/home/radekj/vmmr/datasets"
source = os.path.join(root, "dest240x320")
dest = os.path.join(root, "model_ds")


def get_make_count(path) -> int:
    count = 0
    for model in os.listdir(path):
        model_path = join(path, model)
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            images = list(filter(lambda x: x[-4:] == ".jpg", files))
            count += len(images)

    return count


if __name__ == "__main__":
    makes = sorted(os.listdir(source))
    makes = ["Fiat"]
    for make in makes:
        src_make_path = os.path.join(source, make)
        des_make_path = os.path.join(dest, make)
        if not os.path.exists(des_make_path):
            os.mkdir(des_make_path)

        if os.path.exists(src_make_path) and os.path.isdir(src_make_path):
            models = sorted(os.listdir(src_make_path))
            for name in models:
                src_model_path = os.path.join(src_make_path, name)
                if os.path.exists(src_model_path) and os.path.isdir(src_model_path):
                    parts = name.split("_")
                    if len(parts) >= 2:
                        model = "{}_{}".format(parts[0], parts[1])
                        print(model)
                        des_model_path = os.path.join(des_make_path, model)
                        if not os.path.exists(des_model_path):
                            os.mkdir(des_model_path)

                        for image in os.listdir(src_model_path):
                            src_filename = join(src_model_path, image)
                            dest_filename = join(des_model_path, image)
                            if not os.path.exists(dest_filename):
                                shutil.copy(src_filename, dest_filename)
