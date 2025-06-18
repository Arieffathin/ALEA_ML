from ultralytics import YOLO
import random
import os

# Definisi path absolut
BASE_DIR = r'd:\Skripsi\ModelbaliSudahfix'
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train2', 'weights', 'best.pt')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train', 'images')

# Memastikan file model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model tidak ditemukan di: {MODEL_PATH}')

# Load model hasil training
model = YOLO(MODEL_PATH)

# Pilih gambar acak dari folder train
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f'Folder dataset tidak ditemukan di: {DATASET_PATH}')

all_images = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(root, file))

if not all_images:
    raise FileNotFoundError('Tidak ditemukan gambar di folder train.')

random_img = random.choice(all_images)
print(f'Gambar yang dipilih: {random_img}')

# Inference pada gambar acak
results = model.predict(source=random_img, save=True, conf=0.25)

# Simpan hasil di folder yang spesifik
output_path = os.path.join(BASE_DIR, 'runs', 'detect', 'predict')
print(f'Inference selesai. Hasil prediksi disimpan di: {output_path}')
