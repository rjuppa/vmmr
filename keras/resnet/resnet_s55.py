from __future__ import print_function
import math, json, os, pickle, sys

import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DATADIR = "/storage/plzen1/home/radekj/vmmr"
DATADIR = "/Users/radekj/devroot/vmmr"

name = "sample55"
result_path = os.path.join(DATADIR, 'results')
log_file = "{}/{}/{}_log.csv".format(result_path, name, name)
csv_logger = CSVLogger(log_file, append=True)


SIZE = (224, 224)
BATCH_SIZE = 32
EPOCH = 30

num_classes = 5
input_shape = (224, 224, 3)


def get_model():
    model = keras.applications.resnet50.ResNet50()
    return model


def save_history(history):
    hist_file = "{}/{}/{}_log.json".format(result_path, name, name)
    with open(hist_file, 'w') as file_pi:
        file_pi.write(json.dumps(history.history))


def write_to_log(epoch, logs):
    my_log_file = "{}/{}/{}_log.txt".format(result_path, name, name)
    if not os.path.isfile(my_log_file):
        with open(my_log_file, mode='a+') as f:
            f.write("epoch,loss,acc,val_loss,val_acc,\n")

    with open(my_log_file, mode='a') as f:
        f.write("{}, {}, {}, {}, {},\n".format(
            epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))


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

    shift = 0.05
    train_gen = ImageDataGenerator(
            width_shift_range=shift,
            height_shift_range=shift,
            horizontal_flip=False,
            vertical_flip=True,
            rotation_range=4,
            zoom_range=0.1)
    batches = train_gen.flow_from_directory(directory=TRAIN_DIR,
                                            target_size=SIZE,
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            class_mode="categorical",
                                            shuffle=True,
                                            save_to_dir=save_aug)

    val_gen = ImageDataGenerator()
    val_batches = val_gen.flow_from_directory(directory=VALID_DIR,
                                              target_size=SIZE,
                                              color_mode="rgb",
                                              batch_size=BATCH_SIZE,
                                              class_mode="categorical",
                                              shuffle=True)

    model = get_model()
    classes = list(iter(batches.class_indices))
    model.layers.pop()

    # add last layer
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.summary()
    # opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # opt = Adam(lr=0.0001)
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    finetuned_model.compile(optimizer=opt,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    my_log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: write_to_log(epoch, logs),)
    saved_model = "{}/{}/{}_best.h5".format(result_path, name, name)
    check_pointer = ModelCheckpoint(saved_model, verbose=1, save_best_only=True)
    print("batches.batch_size: {}".format(batches.batch_size))
    print("num_valid_steps: {}".format(num_valid_steps))
    print("num_train_steps: {}".format(num_train_steps))
    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCH,
                                callbacks=[early_stopping, check_pointer, my_log_callback],
                                validation_data=val_batches, validation_steps=num_valid_steps)

    save_history(history)
    saved_model = "{}/{}/{}_final.h5".format(result_path, name, name)
    model.save("{}_final.h5".format(saved_model))


if __name__ == '__main__':
    """
    dataset_path: /Users/radekj/devroot/vmmr/datasets/sample5
    /storage/plzen1/home/radekj/vmmr"

    """
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Need param: python train_vgg16.py dataset_path")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    print("===== name: {}".format(name))
    print("===== folder: {}".format(folder))
    train_vgg(folder)
    print("===== end.")
