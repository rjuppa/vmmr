#!/usr/bin/env python

# this script separates interior and exterior images.
# loading from /source/ dir.
# using Resnet50 model: resnet_model.h5
# using SVM model: svm_model18.sav
# output is saved in /cleaned or /interior folders

import codecs
import json
import os
import pickle
import sys
import time

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# /dir/source/Audi/Audi_A3_2008/45a88c858e.jpg


def copy_file_back(src, is_car):
    # save images
    if is_car:
        target = src.replace("/source/", "/cleaned/")
    else:
        target = src.replace("/source/", "/interier/")

    arr = target.split("/")[:-1]
    path = "/".join(arr)
    if not os.path.exists("/{}".format(path)):
        os.makedirs(path)

    # print("copy {}  ->  {}".format(src, target))
    copyfile(src, target)


def get_model():
    loaded_model = load_model('resnet_model.h5')
    loaded_model.compile(loss='mean_squared_error', optimizer='sgd')
    return loaded_model


def get_svm():
    filename = 'svm_model18.sav'
    return pickle.load(open(filename, 'rb'))


def process_images(folder):
    n = 0
    files = os.listdir(folder)
    print(folder)
    print(len(files))
    files = list(map(lambda x: os.path.join(folder, x), files))
    model = get_model()
    svm = get_svm()
    for f in sorted(files):
        exists = os.path.isfile(f)
        print("{} - {}".format(f, exists))
        if exists and f[-4:] == ".jpg":
            img = image.load_img(f, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # vector on the end of Resnet
            # shape (1, 2048)
            model_output = model.get_layer("avg_pool").output
            intermediate_layer_model = Model(inputs=model.input, outputs=model_output)
            intermediate_output = intermediate_layer_model.predict(x)

            pred = svm.predict(intermediate_output.reshape(1, -1))
            is_car = pred[0] == 1
            n += 1

            # copy an image
            print("{} - {}".format(f[-40:], is_car))
            copy_file_back(f, is_car)


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Need param: python svm_classifier.py path")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    if "/source/" not in folder:
        print("Folder '{}' must be in /source/ directory.".format(folder))
        exit(1)

    # serialize model to JSON
    # model = ResNet50(weights='imagenet')
    # model.save("resnet_model.h5")

    # model_json = model.to_json()
    # with open("resnet_model.json", "w") as json_file:
    #     json_file.write(model_json)

    process_images(folder)
    print("===== end.")
