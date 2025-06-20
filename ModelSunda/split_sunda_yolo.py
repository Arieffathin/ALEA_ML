import os
import shutil
import random


root_src = r'd:/Skripsi/ModelSunda/Aksara-Sunda'
root_dst_img_train = r'd:/Skripsi/ModelSunda/dataset_split/images/train'
root_dst_img_val = r'd:/Skripsi/ModelSunda/dataset_split/images/val'
root_dst_lbl_train = r'd:/Skripsi/ModelSunda/dataset_split/labels/train'
root_dst_lbl_val = r'd:/Skripsi/ModelSunda/dataset_split/labels/val'


class_names = [d for d in os.listdir(root_src) if os.path.isdir(os.path.join(root_src, d))]

for idx, class_name in enumerate(class_names):
    src_dir = os.path.join(root_src, class_name)
    dst_img_train = os.path.join(root_dst_img_train, class_name)
    dst_img_val = os.path.join(root_dst_img_val, class_name)
    dst_lbl_train = os.path.join(root_dst_lbl_train, class_name)
    dst_lbl_val = os.path.join(root_dst_lbl_val, class_name)
    os.makedirs(dst_img_train, exist_ok=True)
    os.makedirs(dst_img_val, exist_ok=True)
    os.makedirs(dst_lbl_train, exist_ok=True)
    os.makedirs(dst_lbl_val, exist_ok=True)


    all_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]  
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]


    for f in train_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_img_train, f))
        with open(os.path.join(dst_lbl_train, f.replace('.jpg', '.txt')), 'w') as label_file:
            label_file.write(f'{idx} 0.5 0.5 1.0 1.0\n')


    for f in val_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_img_val, f))
        with open(os.path.join(dst_lbl_val, f.replace('.jpg', '.txt')), 'w') as label_file:
            label_file.write(f'{idx} 0.5 0.5 1.0 1.0\n')

    print(f"Kelas: {class_name} (id: {idx}) | Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

print('Selesai split dan labeling semua kelas.')
