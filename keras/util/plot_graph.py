import matplotlib
matplotlib.use("TkAgg")   # use for OSX

import math, json, os, pickle, sys

import keras
import matplotlib.pyplot as plt

DATADIR = "/Users/radekj/devroot/vmmr"


def plot_history(filename, header):
    # loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    # val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    # acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    # val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    import pandas
    df = pandas.read_csv(filename, sep=",")
    print(df["loss"])

    epoch_count = len(list(df["loss"]))

    max_loss = list(df["loss"])[-1]
    label_loss = "Training loss ({})".format(max_loss)
    max_val_loss = list(df["val_loss"])[-1]
    label_val_loss = "Validation loss ({})".format(max_val_loss)

    # Loss
    plt.figure(1)
    plt.plot(df["epoch"], df["loss"], 'b', label=label_loss)
    plt.plot(df["epoch"], df["val_loss"], 'g', label=label_val_loss)

    plt.title(header)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    axes = plt.gca()
    xmax = 30
    if xmax < epoch_count:
        xmax = epoch_count
    axes.set_xlim([0, xmax])
    axes.set_ylim([0, 2.0])

    # Accuracy
    max_acc = list(df["acc"])[-1]
    label_acc = "Training accuracy ({})".format(max_acc)
    max_val_acc = list(df["val_acc"])[-1]
    label_val_acc = "Validation accuracy ({})".format(max_val_acc)

    plt.figure(2)
    plt.plot(df["epoch"], df["acc"], 'b', label=label_acc)
    plt.plot(df["epoch"], df["val_acc"], 'g', label=label_val_acc)

    plt.title(header)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    xmax = 30
    if xmax < epoch_count:
        xmax = epoch_count

    axes = plt.gca()
    axes.set_xlim([0, xmax])
    axes.set_ylim([0, 1.0])

    plt.show()


if __name__ == '__main__':
    # filename = 'simple_cnn1.csv'
    path = os.path.join(DATADIR, "results/sample62")
    filename = os.path.join(path, "sample62_resnet256.txt")
    header = "RESNET"
    plot_history(filename, header)
