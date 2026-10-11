"""
data_prep.py - Tải dataset từ Roboflow, gộp và chia train/val cho bài toán
phân loại phương tiện giao thông.

Cấu hình (đường dẫn + API key) lấy từ module config.py thay vì hardcode.
"""
import os
import shutil
import glob
import random
from roboflow import Roboflow

from config import WORKSPACE_DIR, DATASETS_DIR, get_api_key

# API key đọc từ biến môi trường ROBOFLOW_API_KEY (xem .env.example)
API_KEY = get_api_key(required=True)

DATA_DIR = str(DATASETS_DIR)
MERGED_DIR = os.path.join(DATA_DIR, "vietnam_traffic_cls")

os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

print("Initializing Roboflow...")
rf = Roboflow(api_key=API_KEY)

print("Downloading Dataset 1...")
try:
    project1 = rf.workspace("nada-elokaby").project("vehicles-classification-mu804")
    dataset1 = project1.version(1).download("folder")
    ds1_path = os.path.join(DATA_DIR, "vehicles-classification-mu804-1")
    if not os.path.exists(ds1_path):
        paths = glob.glob(os.path.join(DATA_DIR, "vehicles-classification*"))
        ds1_path = paths[0] if paths else None
except Exception as e:
    print("Error DS1:", e)
    ds1_path = None

print("Downloading Dataset 2...")
try:
    project2 = rf.workspace("atul-arun-kanse-njfby").project("vehicle-models-y8tzo")
    dataset2 = project2.version(1).download("folder")
    ds2_path = os.path.join(DATA_DIR, "vehicle-models-y8tzo-1")
    if not os.path.exists(ds2_path):
        paths = glob.glob(os.path.join(DATA_DIR, "vehicle-models*"))
        ds2_path = paths[0] if paths else None
except Exception as e:
    print("Error DS2:", e)
    ds2_path = None

print(f"Path DS1: {ds1_path}")
print(f"Path DS2: {ds2_path}")

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
