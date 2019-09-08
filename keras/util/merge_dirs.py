import os
import shutil

# merge folders of images


def get_name(name):
    new_name = name.strip()
    new_name = new_name.replace(" ", "_")
    new_name = new_name.replace("__", "_")
    new_name = new_name.replace("__", "_")
    return new_name


def check_dir(source, dest):
    for make in sorted(os.listdir(source)):
        src_make_path = os.path.join(source, make)
        des_make_path = os.path.join(dest, make)     # dir/porshe
        print(make)
        if os.path.isdir(src_make_path):
            if not os.path.exists(des_make_path):
                os.mkdir(des_make_path)

            for model in sorted(os.listdir(src_make_path)):
                correct_model = get_name(model)
                src_model_path = os.path.join(src_make_path, model)
                des_model_path = os.path.join(des_make_path, correct_model)
                if not os.path.exists(des_model_path):
                    os.mkdir(des_model_path)

                print(des_model_path)
                for img in os.listdir(src_model_path):
                    src_img_path = os.path.join(src_model_path, img)
                    des_img_path = os.path.join(des_model_path, img)
                    if not os.path.exists(des_img_path):
                        shutil.copy(src_img_path, des_img_path)


if __name__ == '__main__':
    source = "/Users/radekj/devroot/vmmr/datasets/fix1/cleaned"
    dest = "/Users/radekj/devroot/vmmr/datasets/dest"
    check_dir(source, dest)
