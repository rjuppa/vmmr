#!/usr/bin/env python3

# create a dataset with same size of classes
# use augmentation if necessary

import os, math
from os.path import join

import shutil
from util import augment


TOTAL = 5000
MIN_LIMIT = 50
parent_path = os.path.abspath('.')
# root = "/Users/radekj/devroot/vmmr/datasets/"
root = "/storage/plzen1/home/radekj/vmmr/datasets"


def just_copy_files(src_model_path, des_model_path):
    # just copy files
    if not os.path.exists(des_model_path):
        os.mkdir(des_model_path)

    counter = 0
    if os.path.isdir(src_model_path):
        # iterate images in model dir
        for name in os.listdir(src_model_path):
            if name[-4:] == ".jpg":
                src_filename = join(src_model_path, name)
                dest_filename = join(des_model_path, name)
                if not os.path.exists(dest_filename):
                    shutil.copy(src_filename, dest_filename)
                    counter += 1

                if counter > TOTAL:
                    return  # stop


def augment_folder(src_model_path, des_model_path, k):
    if not os.path.exists(des_model_path):
        os.mkdir(des_model_path)

    counter = 0
    if os.path.isdir(src_model_path):
        for name in os.listdir(src_model_path):
            filename = join(src_model_path, name)
            if os.path.isfile(filename) and filename[-4:] == ".jpg":
                # make k-times augmentation
                p = augment.ImageProcessor(src_model_path, name, des_model_path)
                p.mix_it(k-1)
                counter += k
                if counter > TOTAL:
                    return  # stop


if __name__ == "__main__":
    source = os.path.join(root, "model_ds")
    dest = os.path.join(root, "models")
    makes = sorted(os.listdir(source))
    makes = ["Skoda"]
    for make in makes:
        src_make_path = os.path.join(source, make)
        des_make_path = os.path.join(dest, make)
        if not os.path.exists(des_make_path):
            os.mkdir(des_make_path)

        if os.path.exists(src_make_path) and os.path.isdir(src_make_path):
            models = sorted(os.listdir(src_make_path))
            for model in models:
                src_model_path = os.path.join(src_make_path, model)
                des_model_path = os.path.join(des_make_path, model)
                if not os.path.exists(des_model_path):
                    os.mkdir(des_model_path)

                n = len(list(filter(lambda x: x[-4:] == ".jpg", os.listdir(src_model_path))))
                if n < MIN_LIMIT:
                    # skip small classes
                    continue

                k = math.ceil(TOTAL / n)
                print("{}  N={}, K={}".format(model, n, k))

                if k == 1:
                    just_copy_files(src_model_path, des_model_path)
                else:
                    augment_folder(src_model_path, des_model_path, k)
