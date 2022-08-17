import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
import random
from tqdm import tqdm
import os

path = 'data_yolo_format'
labels_path = 'labels_yolo'


def get_all_labels():
    imgs_path_src = 'train/images'
    labels_path_src = 'train/labels'
    labels_map = {'car': 0, 'head': 1, 'face': 2, 'human': 3, 'carplate': 4}
    labels_src = os.listdir(labels_path_src)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    for img_name in tqdm(os.listdir(imgs_path_src)):
        name, _ = os.path.splitext(img_name)
        for class_name in labels_map.keys():
            class_id = labels_map[class_name]
            txt_name = f'{name}_{class_name}..txt'
            if txt_name in labels_src:
                with open(os.path.join(labels_path_src, txt_name), 'r') as reader:
                    lines = reader.readlines()
                    for line in lines:
                        line.rstrip()
                        new_line_arr = line.split(' ')
                        new_line_arr[0] = str(class_id)
                        new_line = ' '.join(new_line_arr)
                        out_label = f'labels_yolo/{name}.txt'
                        with open(out_label, 'a') as wr:
                            wr.write(new_line)


def split_dataset():
    dirs = [path, f'{path}/data',
            f'{path}/data/images', f'{path}/data/labels',
            f'{path}/data/images/train', f'{path}/data/labels/train',
            f'{path}/data/images/test', f'{path}/data/labels/test']
    for di in dirs:
        os.makedirs(di, exist_ok=True)
    # images_names = ['_'.join(file_name.split('_')[:-1]) for file_name in labels_src]
    images_names = os.listdir(labels_path)
    print(len(images_names))
    train_images_names, test_images_names = train_test_split(images_names, test_size=0.07, random_state=42)
    print(len(train_images_names), len(test_images_names))
    for train_image_txt in train_images_names:
        # labels
        name, _ = os.path.splitext(train_image_txt)
        img_name = f"{name}.jpg"
        shutil.copy(labels_path + '/' + train_image_txt, f'{path}/data/labels/train/' + f'{train_image_txt}')
        shutil.copy('train/images/' + img_name, f'{path}/data/images/train/')

    for test_image_txt in test_images_names:
        # labels
        name, _ = os.path.splitext(test_image_txt)
        img_name = f"{name}.jpg"
        shutil.copy(labels_path + '/' + test_image_txt, f'{path}/data/labels/test/' + test_image_txt)
        shutil.copy('train/images/' + img_name, f'{path}/data/images/test/')


if __name__ == "__main__":
    get_all_labels()
    split_dataset()
# pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
