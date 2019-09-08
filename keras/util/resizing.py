#!/usr/bin/env python3

# resize all images in dataset

import os, sys
from os.path import join

import cv2

parent_path = os.path.abspath('.')
root = "/Users/radekj/devroot/vmmr/datasets/"
# root = "/storage/plzen1/home/radekj/vmmr/datasets"
source = os.path.join(root, "dest")
dest = os.path.join(root, "dest240x320")


def resize_folder(src_path, des_path):
    for model in os.listdir(src_path):
        src_model_path = join(src_path, model)
        des_model_path = join(des_path, model)
        if os.path.isdir(src_model_path):
            if not os.path.exists(des_model_path):
                os.mkdir(des_model_path)

            print(model)
            for name in os.listdir(src_model_path):
                filename = join(src_model_path, name)
                if os.path.isfile(filename) and filename[-4:] == ".jpg":
                    image = cv2.imread(filename, cv2.IMREAD_COLOR)
                    resized = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
                    filename = join(des_model_path, name)
                    cv2.imwrite(filename, resized)


if __name__ == "__main__":
    makes = ["Ferrari", "Subaru"]
    for make in makes:
        src_path = os.path.join(source, make)
        des_path = os.path.join(dest, make)
        if os.path.exists(src_path) and os.path.isdir(src_path):

            if not os.path.exists(des_path):
                os.mkdir(des_path)
            resize_folder(src_path, des_path)
