from __future__ import print_function
import math, json, os, pickle, sys

import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = "/storage/plzen1/home/radekj/vmmr"

name = "vgg16"
log_path = os.path.join(DATADIR, 'results')
log_file = "{}/{}/{}_log.csv".format(log_path, name, name)
csv_logger = CSVLogger(log_file, append=True)

my_log_path = os.path.join(DATADIR, 'results')
my_log_file = "{}/{}/{}_log.txt".format(my_log_path, name, name)

SIZE = (224, 224)
BATCH_SIZE = 64
EPOCH = 20

num_classes = 5
input_shape = (224, 224, 3)


def get_model():
    model = keras.applications.vgg16.VGG16()
    return model


def save_history(history):
    hist_path = os.path.join(DATADIR, 'results')
    hist_file = "{}/{}/{}_log.json".format(hist_path, name, name)
    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def write_to_log(epoch, logs):
    if not os.path.isfile(my_log_file):
        with open(my_log_file, mode='a+') as f:
            f.write("epoch, loss, acc\n")

    with open(my_log_file, mode='a') as f:
        # epoch, loss, acc
        f.write("{}, {}, {},\n".format(epoch, logs['loss'], logs['acc']))


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
    for layer in model.layers[:-6]:	# freeze all layers except last 6
        layer.trainable = False
    
    # add last layer
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    finetuned_model.compile(optimizer=rms, loss='categorical_crossentropy',
                            metrics=['accuracy'])

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    my_log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: write_to_log(epoch, logs),)
    check_pointer = ModelCheckpoint("{}_best.h5".format(name), verbose=1, save_best_only=True)
    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCH,
                                            callbacks=[csv_logger, early_stopping, check_pointer, my_log_callback],
                                            validation_data=val_batches,validation_steps=num_valid_steps)
    save_history(history)
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







