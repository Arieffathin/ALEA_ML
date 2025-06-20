from ultralytics import YOLO
import random
import os

model = YOLO('runs/detect/train4/weights/best.pt')
 
folder = r'dataset_split/images/val'
all_classes = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
all_images = []
for class_dir in all_classes:
    all_images += [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.jpg')]

if not all_images:
    raise FileNotFoundError('Tidak ditemukan gambar .jpg di folder train.')

random_img = random.choice(all_images)
print(f'Gambar yang dipilih: {random_img}')

results = model.predict(source=random_img, save=True, conf=0.25)

print('Inference selesai. Hasil gambar dengan bounding box ada di folder runs/detect/predict')
