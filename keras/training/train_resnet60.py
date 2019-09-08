#!/usr/bin/env python3

# import matplotlib
# matplotlib.use("TkAgg")   # use for OSX

import math, json, os, pickle, sys

import keras
# import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator

do_log = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = "/storage/plzen1/home/radekj/vmmr"

name = "sample6"
log_path = os.path.join(DATADIR, 'results')
log_file = "{}/{}/{}_log.csv".format(log_path, name, name)
csv_logger = CSVLogger(log_file, append=True)

my_log_path = os.path.join(DATADIR, 'results')
my_log_file = "{}/{}/{}_log.txt".format(my_log_path, name, name)

SIZE = (224, 224)
BATCH_SIZE = 64
EPOCH = 50

def save_history(history):
    hist_path = os.path.join(DATADIR, 'results')
    hist_file = "{}/{}/{}_log.json".format(hist_path, name, name)
    with open(hist_file, 'w') as file_pi:
        file_pi.write(json.dumps(history.history))


def write_to_log(epoch, logs):
    if not os.path.isfile(my_log_file):
        with open(my_log_file, mode='a+') as f:
            f.write("epoch, loss, acc, val_loss, val_acc\n")

    with open(my_log_file, mode='a') as f:
        # epoch, loss, acc
        f.write("{}, {}, {}, {}, {},\n".format(epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))

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
    gen = ImageDataGenerator(
                             width_shift_range=shift,
                             height_shift_range=shift,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rotation_range=8,
                             zoom_range=0.1)

    # gen = ImageDataGenerator()
    val_gen = ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.resnet50.ResNet50()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False

    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    opt = RMSprop(lr=0.001)
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    finetuned_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=5)
    my_log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: write_to_log(epoch, logs),)
    checkpointer = ModelCheckpoint("{}_best.h5".format(name), verbose=1, save_best_only=True)

    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCH,
                                  callbacks=[early_stopping, checkpointer, my_log_callback],
                                  validation_data=val_batches,
                                  validation_steps=num_valid_steps)

    save_history(history)
    finetuned_model.save("{}_final.h5".format(name))
    finetuned_model.summary()
    # plot_history(history)


if __name__ == '__main__':
    """
    dataset_path: /Users/radekj/devroot/vmmr/datasets/sample5
                  /storage/plzen1/home/radekj/vmmr"
    
    """
    print("===== start")
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Need param: python train_resnet50.py dataset_path")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    exists = os.path.isfile(log_file)
    if not exists:
        f = open(log_file, "w+")
        f.write("====== start ====\n")
        f.close()

    train_cnn(folder)
    print("===== end.")
