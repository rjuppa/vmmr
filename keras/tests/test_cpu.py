import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("===== _start_ ====")
from keras.applications.inception_resnet_v2 import InceptionResNetV2


def print_files(dir_path):
    for f in os.listdir(dir_path):
        print("- {}".format(f))


if __name__ == '__main__':
    print("start")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print("current dir: {}".format(cur_dir))
    print_files(cur_dir)

    print("Loading Keras model InceptionResNetV2.")
    model = InceptionResNetV2()
    print("end.")

