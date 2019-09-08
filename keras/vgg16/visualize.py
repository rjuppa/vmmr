from __future__ import print_function

import os, time
import matplotlib
matplotlib.use("TkAgg")   # use for OSX

import matplotlib.pyplot as plt

import numpy as np
from keras.preprocessing import image
from keras import layers
from keras.applications import vgg16, resnet50
from keras.models import Model, load_model
from keras import backend as K


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

SIZE = (224, 224)
DATADIR = "/Users/radekj/devroot/vmmr/datasets"



def get_model():
    loaded_model = load_model('sample52_best.h5')
    loaded_model.compile(loss='mean_squared_error', optimizer='sgd')
    return loaded_model


if __name__ == '__main__':
    # the name of the layer we want to visualize
    # (see model definition at keras/applications/vgg16.py)
    # load the model
    model = get_model()
    print(model.summary())
    # redefine model to output right after the first hidden layer
    ixs = [2, 5, 9, 13, 17]
    outputs = [model.layers[i].output for i in ixs]
    names = [model.layers[i].name for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)

    # load the image with the required shape
    img_path = os.path.join(DATADIR, "sample52/valid/Audi/596d864085db9ac15fea440d3541fd87b4f8b6c0.jpg")
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    # get feature map for first hidden layer
    feature_maps = model.predict(img)

    square = 8
    n = 0
    for fmap in feature_maps:

        for i in range(16):
            plt.matshow(fmap[0, :, :, i], cmap='viridis')
            pyplot.show()

        # plot all 64 maps in an 8x8 squares
        ix = 1
        print(names[n])
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(fmap[0, :, :, ix-1], cmap='viridis')
                ix += 1
        # show the figure
        n += 1
        pyplot.show()

