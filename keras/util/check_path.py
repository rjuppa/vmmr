import os, sys
import shutil


def check_name(name):
    new_name = name.strip()
    new_name = new_name.replace(" ", "_")
    new_name = new_name.replace("__", "_")
    new_name = new_name.replace("__", "_")
    return new_name


def copy_files(src, des):
    if not os.path.exists(des):
        os.mkdir(des)

    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        des_path = os.path.join(des, name)
        if os.path.isfile(src_path) and src_path[-4:] == ".jpg":
            if not os.path.exists(des_path):
                shutil.copy(src_path, des_path)

            if os.path.exists(des_path) and os.path.isfile(des_path):
                if os.path.getsize(des_path) > 0:
                    # remove source file if dest exists
                    os.remove(src_path)


def check_dir(path):
    for name in os.listdir(path):
        # makes
        dir_make = os.path.join(path, name)
        print(name)
        if os.path.exists(dir_make) and os.path.isdir(dir_make):
            # models
            for model in os.listdir(dir_make):
                correct_model = check_name(model)
                if correct_model != model:
                    # only if it doesnt match
                    src_path = os.path.join(dir_make, model)
                    print(src_path)
                    des_path = os.path.join(dir_make, correct_model)
                    copy_files(src_path, des_path)

                    if len(os.listdir(src_path)) == 0:
                        shutil.rmtree(src_path)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Need param: python gen_path.py '/vmmr/datasets/source'")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    if "/datasets/" not in folder:
        print("Folder '{}' must be in /datasets/ directory.".format(folder))
        exit(1)

    check_dir(folder)
