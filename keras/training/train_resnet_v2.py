#!/usr/bin/env python3

# import matplotlib
# matplotlib.use("TkAgg")   # use for OSX

import math, json, os, pickle, sys

import keras
# import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

do_log = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

name = "ResNetV2"
log_file = "{}_history_log.csv".format(name)
DIRLOG = "/storage/plzen1/home/radekj/vmmr/results/resnet_v2"
csv_logger = CSVLogger(os.path.join(DIRLOG, log_file), append=True)


SIZE = (224, 224)
BATCH_SIZE = 64
EPOCH = 20


def train_cnn(folder):
    DATA_DIR = folder
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    save_aug = os.path.join(DATA_DIR, 'tmp')

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    shift = 0.05
    # gen = ImageDataGenerator(zca_whitening=True,
    #                          width_shift_range=shift,
    #                          height_shift_range=shift,
    #                          horizontal_flip=True,
    #                          vertical_flip=False,
    #                          rotation_range=8,
    #                          zoom_range=0.1,
    #                          featurewise_center=True,
    #                          featurewise_std_normalization=True)

    gen = ImageDataGenerator()
    val_gen = ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.inception_resnet_v2.InceptionResNetV2()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False

    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=5)
    checkpointer = ModelCheckpoint("{}_best.h5".format(name), verbose=1, save_best_only=True)

    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCH,
                                  callbacks=[csv_logger, early_stopping, checkpointer],
                                  validation_data=val_batches,
                                  validation_steps=num_valid_steps)

    finetuned_model.save("{}_final.h5".format(name))


if __name__ == '__main__':
    """
    dataset_path: /Users/radekj/devroot/vmmr/datasets/sample5
                  /storage/plzen1/home/radekj/vmmr"
    
    """

    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Need param: python train.py dataset_path")
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

    train_cnn(folder)
    print("===== end.")
