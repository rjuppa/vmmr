#!/usr/bin/env python3

# create a dataset with same size of classes
# use augmentation if necessary

import os, math
from os.path import join

import cv2
import numpy as np
import random
import shutil


TOTAL = 5000
MIN_LIMIT = 50
parent_path = os.path.abspath('.')
# root = "/Users/radekj/devroot/vmmr/datasets/"
root = "/storage/plzen1/home/radekj/vmmr/datasets"


class ImageProcessor:
    image = None
    original = None
    width = 320
    height = 240
    mark = ""
    filters = ["lighter", "saturation", "blur", "invert", "contrast",
               "affinet", "grayscale", "hist", "foggy", "rainy", "drops"]

    def __init__(self, folder, filename, des_path):
        self.original = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
        self.image = self.original
        self.folder = folder
        self.filename = filename[:-4] + "_"
        self.origname = filename[:-4] + "_"
        self.des_path = des_path

        path = os.path.join(self.des_path, filename)
        cv2.imwrite(path, self.original)

    def rnd(self, max):
        return random.randint(1, max)

    def get_rand_filter(self, used=None):
        if used is None:
            used = []

        ff = self.filters.copy()
        for f in used:
            ff.remove(f)

        idx = random.randint(1, len(ff))
        return ff[idx-1]

    def mix_it(self, times):
        # iterates filters, use them
        # and apply other filters to augment an image
        counter = 0
        for name in self.filters:
            f1 = name
            used = []
            self.image = self.original
            self.filename = self.origname
            for i in range(7):
                result = self.apply_filter(f1)      # image saved
                counter += 1
                if counter >= times:
                    return  # stop

                self.image = result
                used.append(f1)
                f1 = self.get_rand_filter(used)

    def apply_filter(self, filter_name):
        if not hasattr(self, filter_name):
            raise AttributeError("filter_name")

        method = getattr(self, filter_name)
        result = method()
        self.filename = self.filename + self.mark
        path = os.path.join(self.des_path, self.filename + ".jpg")
        cv2.imwrite(path, result)
        return result

    def lighter(self):
        self.mark = "L"
        table = np.array([((i / 255.0) ** 0.8) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image, table)

    def saturation(self):
        self.mark = "S"
        saturation = 25
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        v = image[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image[:, :, 2] = v
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    def blur(self):
        self.mark = "B"
        return cv2.blur(self.image, (3, 3))

    def invert(self):
        self.mark = "I"
        return cv2.bitwise_not(self.image)

    def contrast(self):
        self.mark = "C"
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def affinet(self):
        self.mark = "A"
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[50, 45], [195, 45], [55, 205]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(self.image, M, (320, 240))

    def grayscale(self):
        self.mark = "G"
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def hist(self):
        self.mark = "H"
        h, s, v = cv2.split(cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV))
        eq_v = cv2.equalizeHist(v)
        return cv2.cvtColor(cv2.merge([h, s, eq_v]), cv2.COLOR_HSV2RGB)

    def foggy(self):
        # foggy.jpg => 600x400
        self.mark = "F"
        x = random.randint(1, 600-self.width)
        y = random.randint(1, 400-self.height)
        alpha = 0.5
        fog = cv2.imread(os.path.join(root, "foggy.jpg"), cv2.IMREAD_COLOR)
        cropped = fog[y:y + self.height, x:x + self.width]
        beta = (1.0 - alpha)
        return cv2.addWeighted(self.image, alpha, cropped, beta, 0.0)

    def rainy(self):
        # rainy.jpg => 600x400
        self.mark = "R"
        x = random.randint(1, 600-self.width)
        y = random.randint(1, 400-self.height)
        alpha = 0.7
        fog = cv2.imread(os.path.join(root, "rainy.jpg"), cv2.IMREAD_COLOR)
        cropped = fog[y:y + self.height, x:x + self.width]
        beta = (1.0 - alpha)
        return cv2.addWeighted(self.image, alpha, cropped, beta, 0.0)

    def drops(self):
        # rainy.jpg => 600x400
        self.mark = "D"
        x = random.randint(1, 600-self.width)
        y = random.randint(1, 400-self.height)
        alpha = 0.9
        fog = cv2.imread(os.path.join(root, "drops.jpg"), cv2.IMREAD_COLOR)
        cropped = fog[y:y + self.height, x:x + self.width]
        beta = (1.0 - alpha)
        return cv2.addWeighted(self.image, alpha, cropped, beta, 0.0)


def just_copy_files(src_path, des_path):
    # take 5 from every model directory
    # and repeat until it is enough
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    counter = 0
    while counter < TOTAL:
        for model in os.listdir(src_path):
            src_model_path = join(src_path, model)
            if os.path.isdir(src_model_path):

                ii = 0
                # iterate images in model dir
                for name in os.listdir(src_model_path):
                    filename = join(src_model_path, name)
                    dest_filename = join(des_path, name)
                    if not os.path.exists(dest_filename):
                        shutil.copy(filename, dest_filename)
                        ii += 1
                        counter += 1

                    if counter > TOTAL:
                        return  # stop

                    if ii > 5:
                        break   # go to next directory


def augment_folder(src_path, des_path, k):
    if not os.path.exists(des_path):
        os.mkdir(des_path)

    counter = 0
    for model in os.listdir(src_path):
        src_model_path = join(src_path, model)
        if os.path.isdir(src_model_path):

            for name in os.listdir(src_model_path):
                filename = join(src_model_path, name)
                if os.path.isfile(filename) and filename[-4:] == ".jpg":
                    # make k-times augmentation
                    p = ImageProcessor(src_model_path, name, des_path)
                    p.mix_it(k-1)

                    counter += k
                    if counter > TOTAL:
                        return  # stop


def get_make_count(path) -> int:
    count = 0
    for model in os.listdir(path):
        model_path = join(path, model)
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            images = list(filter(lambda x: x[-4:] == ".jpg", files))
            count += len(images)

    return count


if __name__ == "__main__":

    source = os.path.join(root, "dest240x320")
    dest = os.path.join(root, "ds5000")
    makes = sorted(os.listdir(source))
    # makes = ["Ferrari", "Fiat", "Subaru"]
    for make in makes:
        src_make_path = os.path.join(source, make)
        des_make_path = os.path.join(dest, make)

        if os.path.exists(src_make_path) and os.path.isdir(src_make_path):
            n = get_make_count(src_make_path)
            if n < MIN_LIMIT:
                # skip small classes
                continue

            k = math.ceil(TOTAL / n)
            print("{}  N={}, K={}".format(make, n, k))

            if k == 1:
                just_copy_files(src_make_path, des_make_path)
            else:
                augment_folder(src_make_path, des_make_path, k)
