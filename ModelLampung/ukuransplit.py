import os
import pandas as pd

def count_images_per_class(base_dir):
    data = []
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                n_files = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                data.append({'class': class_name, 'data_type': split, 'count': n_files})
    df = pd.DataFrame(data)
    if not df.empty:
        pivot = df.pivot(index='class', columns='data_type', values='count').fillna(0).astype(int)
        print(pivot)
        print("\nUkuran total per split:")
        print(pivot.sum())
    else:
        print('Tidak ada data ditemukan.')

if __name__ == "__main__":
    count_images_per_class("dataset_split/images")