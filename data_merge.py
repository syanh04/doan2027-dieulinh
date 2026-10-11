"""
data_merge.py - Gộp 2 dataset phân loại (Roboflow) thành 1 dataset cân bằng
và chia train/val theo tỉ lệ 70:30.

Đường dẫn lấy từ module config.py (tự suy ra theo vị trí file).
"""
import os
import shutil
import random

from config import DATASETS_DIR

DATA_DIR = os.path.join(str(DATASETS_DIR))
MERGED_DIR = os.path.join(DATA_DIR, "vietnam_traffic_cls")

ds1_path = os.path.join(DATA_DIR, "vehicles-classification-1")
ds2_path = os.path.join(DATA_DIR, "Vehicle-Models-1")

def get_target_class_name(class_name):
    name = class_name.lower()
    if 'motor' in name or 'bike' in name: return '0_Xe_may'
    if 'car' in name or 'suv' in name or 'taxi' in name: return '1_O_to_con'
    if 'bus' in name or 'coach' in name: return '2_Xe_khach'
    if 'truck' in name or 'van' in name or 'pickup' in name or 'lgv' in name or 'hgv' in name: return '3_Xe_tai'
    return '1_O_to_con'

all_images = []

def process_classification_folder(ds_path):
    if not ds_path or not os.path.exists(ds_path): return
    for split in ['train', 'valid', 'val', 'test']:
        split_dir = os.path.join(ds_path, split)
        if not os.path.exists(split_dir): continue
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            if not os.path.isdir(class_path): continue
            target_class = get_target_class_name(class_folder)
            for img_file in os.listdir(class_path):
                if not (img_file.endswith('.jpg') or img_file.endswith('.png') or img_file.endswith('.jpeg')): continue
                src_img = os.path.join(class_path, img_file)
                all_images.append((src_img, target_class))

process_classification_folder(ds1_path)
process_classification_folder(ds2_path)

print(f"Total images collected: {len(all_images)}")

random.shuffle(all_images)
train_count = int(len(all_images) * 0.7)

print(f"Split Train: {train_count}, Val: {len(all_images) - train_count}")

for i, (img_path, target_class) in enumerate(all_images):
    split = 'train' if i < train_count else 'val'
    dst_dir = os.path.join(MERGED_DIR, split, target_class)
    os.makedirs(dst_dir, exist_ok=True)
    dst_img = os.path.join(dst_dir, f"img_{i}_{os.path.basename(img_path)}")
    shutil.copy2(img_path, dst_img)

print("Done! Data merged for classification.")
