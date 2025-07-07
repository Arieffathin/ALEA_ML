import os
import shutil
import random


SOURCE_DIR = r'd:/Skripsi/ModelLampung/Aksara-Lampung'
OUTPUT_IMG_TRAIN = r'd:/Skripsi/ModelLampung/dataset_split/images/train'
OUTPUT_IMG_VAL = r'd:/Skripsi/ModelLampung/dataset_split/images/val'
OUTPUT_LBL_TRAIN = r'd:/Skripsi/ModelLampung/dataset_split/labels/train'
OUTPUT_LBL_VAL = r'd:/Skripsi/ModelLampung/dataset_split/labels/val'


TRAIN_RATIO = 0.8  

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for idx, class_name in enumerate(class_names):
    src_dir = os.path.join(SOURCE_DIR, class_name)
    dst_img_train = os.path.join(OUTPUT_IMG_TRAIN, class_name)
    dst_img_val = os.path.join(OUTPUT_IMG_VAL, class_name)
    dst_lbl_train = os.path.join(OUTPUT_LBL_TRAIN, class_name)
    dst_lbl_val = os.path.join(OUTPUT_LBL_VAL, class_name)
    make_dir(dst_img_train)
    make_dir(dst_img_val)
    make_dir(dst_lbl_train)
    make_dir(dst_lbl_val)


    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)
    split_idx = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]


    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_img_train, f))
        label_path = os.path.join(dst_lbl_train, os.path.splitext(f)[0] + '.txt')
        with open(label_path, 'w') as label_file:
            label_file.write(f'{idx} 0.5 0.5 1.0 1.0\n')
    for f in val_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_img_val, f))
        label_path = os.path.join(dst_lbl_val, os.path.splitext(f)[0] + '.txt')
        with open(label_path, 'w') as label_file:
            label_file.write(f'{idx} 0.5 0.5 1.0 1.0\n')


    print(f"Kelas: {class_name} | Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

print('Selesai split semua kelas!')
