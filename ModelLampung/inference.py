from ultralytics import YOLO
import random
import os

# Load model hasil training
model = YOLO('runs/detect/train3/weights/modellampung.pt')

# Pilih gambar acak dari folder train (mendukung .jpg, .png, .jpeg)
folder = r'dataset_split/images/train'
all_classes = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
all_images = []
for class_dir in all_classes:
    all_images += [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not all_images:
    raise FileNotFoundError('Tidak ditemukan gambar .jpg/.png/.jpeg di folder train.')

random_img = random.choice(all_images)
print(f'Gambar yang dipilih: {random_img}')

# Inference pada gambar acak, hasil disimpan di folder khusus
results = model.predict(source=random_img, save=True, project='runs/detect/inferensi', name='', exist_ok=True, conf=0.25)

# Tampilkan hasil prediksi di terminal
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'Prediksi: class={cls}, confidence={conf:.2f}')

print('Inference selesai. Hasil gambar dengan bounding box ada di folder runs/detect/inferensi')
