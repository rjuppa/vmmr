#!/usr/bin/env python3

# create a dataset with same size of classes
# use augmentation if necessary

import os

from PIL import Image


# root = "/Users/radekj/devroot/vmmr/datasets/"
root = "/storage/plzen1/home/radekj/vmmr/datasets"


if __name__ == "__main__":
    source = os.path.join(root, "models")
    makes = sorted(os.listdir(source))
    for make in makes:
        src_make_path = os.path.join(source, make)

        if os.path.exists(src_make_path) and os.path.isdir(src_make_path):
            models = sorted(os.listdir(src_make_path))
            for model in models:
                src_model_path = os.path.join(src_make_path, model)
                if os.path.exists(src_model_path) and os.path.isdir(src_model_path):

                    for image in sorted(os.listdir(src_model_path)):
                        file_path = os.path.join(src_model_path, image)
                        if image[-4:] == ".jpg":
                            if os.path.isfile(file_path):

                                try:
                                    img = Image.open(file_path)

                                except Exception as e:
                                    print("{} <= will be deleted".format(file_path))
                                    # os.unlink(file_path)
