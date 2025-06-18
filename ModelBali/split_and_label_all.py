import os
import shutil
import random
from PIL import Image
import yaml

def create_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def calculate_bbox(image_path):
    # Open image and get its size
    with Image.open(image_path) as img:
        width, height = img.size
    
    # For this dataset, we'll consider the character takes up about 80% of the image
    # You might want to adjust these values based on your specific dataset
    bbox_width = 0.8
    bbox_height = 0.8
    
    # Center coordinates
    center_x = 0.5
    center_y = 0.5
    
    return center_x, center_y, bbox_width, bbox_height

def main():
    # Path configuration
    root_src = r'd:\Skripsi\Modelbalisudahfix\Aksara-Bali'
    dataset_root = r'd:\Skripsi\Modelbalisudahfix\dataset'
    
    # Create dataset structure
    train_img_dir = os.path.join(dataset_root, 'train', 'images')
    val_img_dir = os.path.join(dataset_root, 'val', 'images')
    train_label_dir = os.path.join(dataset_root, 'train', 'labels')
    val_label_dir = os.path.join(dataset_root, 'val', 'labels')
    
    # Create all necessary directories
    create_dirs([train_img_dir, val_img_dir, train_label_dir, val_label_dir])
    
    # Get all class names (subfolders)
    class_names = [d for d in os.listdir(root_src) if os.path.isdir(os.path.join(root_src, d))]
    class_names.sort()  # Ensure consistent class ordering
    
    # Create class mapping
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    
    # Process each class
    for class_name, idx in class_dict.items():
        src_dir = os.path.join(root_src, class_name)
        print(f"\nProcessing class: {class_name} (id: {idx})")
        
        try:
            # Get all image files
            all_files = [f for f in os.listdir(src_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not all_files:
                print(f"Warning: No images found in {class_name}")
                continue
                
            # Shuffle and split
            random.shuffle(all_files)
            split_idx = int(len(all_files) * 0.8)  # 80% for training
            train_files = all_files[:split_idx]
            val_files = all_files[split_idx:]
            
            # Process training files
            for f in train_files:
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(train_img_dir, f"{class_name}_{f}")
                
                # Copy image
                shutil.copy2(src_path, dst_path)
                
                # Create label
                center_x, center_y, bbox_w, bbox_h = calculate_bbox(src_path)
                label_path = os.path.join(train_label_dir, f"{class_name}_{os.path.splitext(f)[0]}.txt")
                with open(label_path, 'w') as label_file:
                    label_file.write(f"{idx} {center_x} {center_y} {bbox_w} {bbox_h}\n")
            
            # Process validation files
            for f in val_files:
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(val_img_dir, f"{class_name}_{f}")
                
                # Copy image
                shutil.copy2(src_path, dst_path)
                
                # Create label
                center_x, center_y, bbox_w, bbox_h = calculate_bbox(src_path)
                label_path = os.path.join(val_label_dir, f"{class_name}_{os.path.splitext(f)[0]}.txt")
                with open(label_path, 'w') as label_file:
                    label_file.write(f"{idx} {center_x} {center_y} {bbox_w} {bbox_h}\n")
            
            print(f"Class: {class_name} | Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")
            
        except Exception as e:
            print(f"Error processing class {class_name}: {str(e)}")
    
    # Create data.yaml file
    yaml_data = {
        'path': dataset_root,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(dataset_root, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)
    
    print('\nDataset preparation completed successfully!')
    print(f'Total number of classes: {len(class_names)}')
    print(f'Dataset location: {dataset_root}')
    print('data.yaml file has been created with class mappings')

if __name__ == '__main__':
    main()
