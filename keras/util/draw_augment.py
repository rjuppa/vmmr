#!/usr/bin/env python3

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


img = load_img("/Users/radekj/devroot/vmmr/datasets/skoda.jpg")


if __name__ == "__main__":
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    shift = 0.05
    # datagen = ImageDataGenerator(width_shift_range=[-shift, shift])
    datagen = ImageDataGenerator(rotation_range=10)

    it = datagen.flow(samples, batch_size=1)

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        pyplot.imshow(image)

    pyplot.show()
