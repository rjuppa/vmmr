#!/usr/bin/env python3

# counts images for statistics
# used folder /ds5000

import os

from os import listdir
from os.path import join, isdir


parent_path = os.path.abspath('.')
# root = "/Users/radekj/devroot/vmmr/datasets/"
root = "/storage/plzen1/home/radekj/vmmr/datasets/"
path = os.path.join(root, "ds5000")


def get_make_count(path):
    image_count = 0
    for name in os.listdir(path):
        sub_path = join(path, name)
        if os.path.isfile(sub_path) and sub_path[-4:] == ".jpg":
            image_count += 1

    return image_count


if __name__ == "__main__":
    print("Make, Count")
    counts = {}
    for make in sorted(listdir(path)):
        make_path = join(path, make)
        # print(sub_path)
        if isdir(make_path):
            count = get_make_count(make_path)
            print("{},{}".format(make, count))
