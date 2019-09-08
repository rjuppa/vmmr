from __future__ import print_function
import math, json, os, pickle, sys

import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = "/storage/plzen1/home/radekj/vmmr"

name = "vgg16x"
log_file = "{}_history_log.csv".format(name)
csv_logger = CSVLogger(log_file, append=True)


SIZE = (224, 224)
BATCH_SIZE = 64
EPOCH = 10

num_classes = 5
input_shape = (224, 224, 3)


def get_model():
    model = keras.applications.vgg16.VGG16()
    return model


def train_vgg(folder):
    DATA_DIR = folder
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    save_aug = os.path.join(DATA_DIR, 'tmp')

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])
    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    train_gen = ImageDataGenerator()
    batches = train_gen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = ImageDataGenerator()
    val_batches = val_gen.flow_from_directory(
        directory=VALID_DIR,
        target_size=SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    model = get_model()
    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    finetuned_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                            metrics=['accuracy'])

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    check_pointer = ModelCheckpoint("{}_best.h5".format(name), verbose=1, save_best_only=True)
    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCH,
                                            callbacks=[csv_logger, early_stopping, check_pointer],
                                            validation_data=val_batches, validation_steps=num_valid_steps)
    model.save("{}_final.h5".format(name))


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Need param: python train_vgg16.py dataset_path")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    exists = os.path.isfile(log_file)

    if not exists:
        f = open(log_file, "w+")
        f.write("====== start ====")
        f.close()

    print("===== folder: {}".format(folder))
    train_vgg(folder)
    print("===== end.")







