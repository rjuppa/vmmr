#!/usr/bin/env python3

# this will train SVC classifier
# from directory dir8
# save the model for future
import codecs
import json
import os
import pickle
import sys
import time

import keras

from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model, load_model
import numpy as np
from sklearn.svm import SVC
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# /dir/source/Audi/Audi_A3_2008/45a88c858e.jpg
DATADIR = "/Users/radekj/devroot/vmmr"

name = "dir8"
result_path = os.path.join(DATADIR, 'results')
output = os.path.join(result_path, name)


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

    print("copy {}  ->  {}".format(src, target))
    copyfile(src, target)


def get_model():
    model = keras.applications.vgg16.VGG16()
    return model


def get_svm():
    path = os.path.join(output, "%s.pickle".format(name))
    return pickle.load(open(path, 'rb'))


def process_images(folder):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # svm = get_svm()
    model = get_model()
    sub_folder = os.path.join(folder, "train")
    classes = [c for c in os.listdir(sub_folder) if c[0] != "."]
    for class_name in sorted(classes):
        class_dir = os.path.join(sub_folder, class_name)
        files = os.listdir(class_dir)
        files = list(map(lambda x: os.path.join(class_dir, x), files))
        for f in sorted(files):
            exists = os.path.isfile(f)
            if exists and f[-4:] == ".jpg":
                img = image.load_img(f, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = vgg16.preprocess_input(x)

                # vector on the end of Resnet
                # shape (1, 2048)
                model_output = model.get_layer("fc2").output
                intermediate_layer_model = Model(inputs=model.input, outputs=model_output)
                intermediate_output = intermediate_layer_model.predict(x)
                v = intermediate_output.reshape(1, -1)[0]
                X_train.append(v)
                y_train.append(class_name)

    sub_folder = os.path.join(folder, "valid")
    classes = [c for c in os.listdir(sub_folder) if c[0] != "."]
    for class_name in sorted(classes):
        class_dir = os.path.join(sub_folder, class_name)
        files = os.listdir(class_dir)
        files = list(map(lambda x: os.path.join(class_dir, x), files))
        for f in sorted(files):
            exists = os.path.isfile(f)
            if exists and f[-4:] == ".jpg":
                img = image.load_img(f, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = vgg16.preprocess_input(x)

                # vector on the end of Resnet
                # shape (1, 2048)
                model_output = model.get_layer("fc2").output
                intermediate_layer_model = Model(inputs=model.input, outputs=model_output)
                intermediate_output = intermediate_layer_model.predict(x)
                v = intermediate_output.reshape(1, -1)[0]
                X_test.append(v)
                y_test.append(class_name)

    # training a linear SVM classifier
    svm_model_linear = SVC(kernel='linear', C=8)
    svm_model_linear.fit(X_train, y_train)

    kernel_svm_score = svm_model_linear.score(X_test, y_test)
    print(kernel_svm_score)

    filename = "{}.pickle".format(name)
    pickle.dump(svm_model_linear, open(filename, 'wb'))

    # svm_predictions = svm_model_linear.predict(X_test)

    # pred = svm.predict(intermediate_output.reshape(1, -1))
    # print("pred: {}".format(pred))
    # is_front = pred[0] == 1

    # copy an image
    # print("{} - {}".format(f[-40:], is_front))
    # copy_file_back(f, is_car)


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

    process_images(folder)
    print("===== end.")
