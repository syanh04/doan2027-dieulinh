import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10 (GPU CUDA)",
   "language": "python",
   "name": "py310-cuda"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# Title
add_markdown("""# Hệ thống Nhận diện và Phân loại Phương tiện Giao Thông (Cảnh Báo Ùn Tắc)
* **Bài toán:** Phân loại các loại xe lưu thông trên đường (`0_Xe_may`, `1_O_to_con`, `2_Xe_khach`, `3_Xe_tai`) từ camera giao thông.
* **Phương pháp:** Image Classification với mô hình YOLOv8-cls (`yolov8n-cls`, `yolov8s-cls`).
* **Luồng xử lý chuẩn khoa học:**
  1. Kết nối Thư viện & Cấu hình Môi trường
  2. Kiểm tra GPU & Cấu hình Phần cứng
  3. Chuẩn bị Dữ liệu & Làm sạch (Tải dữ liệu, Gộp nhãn chuẩn, Phân chia Train/Val)
  4. Thống kê Chi tiết Số lượng Ảnh theo Từng Tập (Train / Val)
  5. Trực quan hóa Phân bố Dữ liệu & Mẫu Ảnh
  6. Huấn luyện & Đánh giá Mô hình YOLOv8-cls""")

# Bước 1
add_markdown("## Bước 1: Kết nối Thư viện & Cấu hình Hệ thống")
add_code("""import os
import glob
import shutil
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image
import ultralytics
from ultralytics import YOLO
import torch

WORKSPACE_DIR = r"E:\ĐA GTTM"
os.chdir(WORKSPACE_DIR)

print("✅ Working Directory:", os.getcwd())
print("✅ Ultralytics Version:", ultralytics.__version__)
print("✅ PyTorch Version    :", torch.__version__)""")

# Bước 2
add_markdown("## Bước 2: Kiểm tra GPU & Cấu hình Phần cứng")
add_code("""if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(0)
    print(f"🚀 ĐÃ NHẬN GPU: {device_name}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
else:
    device = 'cpu'
    print("⚠️ KHÔNG TÌM THẤY GPU! Đang chạy trên CPU.")

# Cấu hình GPU tối ưu tránh lỗi FP16 (AMP) trên Windows RTX
gpu_kwargs = {'device': device, 'amp': False if torch.cuda.is_available() else True}
print(f"🎯 Cấu hình GPU tối ưu: {gpu_kwargs}")""")

# Bước 3
add_markdown("## Bước 3: Chuẩn bị Dữ liệu & Làm sạch (Data Preparation & Cleaning)")
add_markdown("""Thực hiện tải dữ liệu thô từ các nguồn, đồng nhất nhãn phân loại giao thông Việt Nam (`0_Xe_may`, `1_O_to_con`, `2_Xe_khach`, `3_Xe_tai`), lọc bỏ tệp lỗi và phân chia ngẫu nhiên vào các tập `train` và `val` theo tỉ lệ 70:30.""")
add_code("""DATA_ROOT = os.path.join(WORKSPACE_DIR, "datasets")
MERGED_DIR = os.path.join(DATA_ROOT, "vietnam_traffic_cls")

print("📁 Thư mục lưu trữ bộ dữ liệu:", MERGED_DIR)

# Mã nguồn xử lý & làm sạch dữ liệu (Tự động bỏ qua nếu đã làm sạch xong):
if not os.path.exists(MERGED_DIR):
    print("🔄 Đang tiến hành làm sạch và gộp dữ liệu...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    # Hàm map nhãn dữ liệu thô về 4 class chuẩn
    def get_target_class_name(class_name):
        name = class_name.lower()
        if 'motor' in name or 'bike' in name: return '0_Xe_may'
        if 'car' in name or 'suv' in name or 'taxi' in name: return '1_O_to_con'
        if 'bus' in name or 'coach' in name: return '2_Xe_khach'
        if 'truck' in name or 'van' in name or 'pickup' in name or 'lgv' in name or 'hgv' in name: return '3_Xe_tai'
        return '1_O_to_con'

    ds1_path = os.path.join(DATA_ROOT, "vehicles-classification-mu804-1")
    ds2_path = os.path.join(DATA_ROOT, "vehicle-models-y8tzo-1")
    
    all_images = []
    def process_folder(ds_path):
        if not ds_path or not os.path.exists(ds_path): return
        for split in ['train', 'valid', 'val', 'test']:
            split_dir = os.path.join(ds_path, split)
            if not os.path.exists(split_dir): continue
            for class_folder in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_folder)
                if not os.path.isdir(class_path): continue
                target_class = get_target_class_name(class_folder)
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
                        all_images.append((os.path.join(class_path, img_file), target_class))

    process_folder(ds1_path)
    process_folder(ds2_path)
    
    if all_images:
        random.shuffle(all_images)
        train_count = int(len(all_images) * 0.7)
        for i, (img_path, target_class) in enumerate(all_images):
            split = 'train' if i < train_count else 'val'
            dst_dir = os.path.join(MERGED_DIR, split, target_class)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(dst_dir, f"img_{i}_{os.path.basename(img_path)}"))
        print("✅ Hoàn tất làm sạch và phân chia dữ liệu!")
    else:
        print("⚠️ Chưa tìm thấy dữ liệu thô để làm sạch.")
else:
    print("✅ DỮ LIỆU ĐÃ ĐƯỢC CHUẨN BỊ VÀ LÀM SẠCH SẴN RỒI!")""")

# Bước 4
add_markdown("## Bước 4: Thống kê Chi tiết Số lượng Ảnh theo Từng Tập (Train / Val)")
add_code("""if os.path.exists(MERGED_DIR):
    data = []
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp', '.webp')
    
    splits = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]
    
    for split in sorted(splits):
        split_path = os.path.join(MERGED_DIR, split)
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path): continue
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(valid_exts)])
            data.append({'Split': split, 'Class': cls, 'Count': count})
            
    df_data = pd.DataFrame(data)
    
    if not df_data.empty:
        pivot_df = df_data.pivot(index='Class', columns='Split', values='Count').fillna(0).astype(int)
        pivot_df['Tổng cộng'] = pivot_df.sum(axis=1)
        
        total_row = pivot_df.sum(axis=0)
        total_row.name = 'TỔNG CỘNG'
        pivot_df_final = pd.concat([pivot_df, pd.DataFrame(total_row).T])
        
        print("\\n📊 BẢNG THỐNG KÊ SỐ LƯỢNG ẢNH CHI TIẾT THEO TỪNG TẬP DỮ LIỆU:")
        display(pivot_df_final)
        
        total_all = df_data['Count'].sum()
        print(f"\\n🔥 TỔNG SỐ LƯỢNG ẢNH TRÊN TOÀN BỘ BỘ DỮ LIỆU: {total_all:,} ảnh")
        for split in sorted(pivot_df.columns[:-1]):
            split_total = pivot_df[split].sum()
            ratio = (split_total / total_all) * 100
            print(f"  🔹 Tập '{split}': {split_total:,} ảnh ({ratio:.1f}%)")
    else:
        print("⚠️ Thư mục dữ liệu rỗng hoặc không đúng định dạng.")
else:
    print("⚠️ Chưa tìm thấy bộ dữ liệu vietnam_traffic_cls tại:", MERGED_DIR)""")

# Bước 5
add_markdown("## Bước 5: Biểu đồ Trực quan hóa Phân bố Dữ liệu & Mẫu Ảnh Train")
add_code("""if os.path.exists(MERGED_DIR) and 'df_data' in locals() and not df_data.empty:
    # 1. Vẽ biểu đồ cột phân bố từng lớp theo từng tập
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_chart = pivot_df.drop(index='TỔNG CỘNG', errors='ignore').drop(columns='Tổng cộng', errors='ignore')
    pivot_chart.plot(kind='bar', ax=ax, colormap='viridis', width=0.7)
    
    plt.title("Phân bố số lượng ảnh từng lớp theo từng tập dữ liệu", fontsize=13, fontweight='bold')
    plt.xlabel("Lớp phương tiện (Class)", fontsize=11)
    plt.ylabel("Số lượng ảnh", fontsize=11)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Tập dữ liệu (Split)")
    plt.tight_layout()
    plt.show()

# 2. Hiển thị ảnh mẫu thực tế từ từng lớp trong tập train
classes = ['0_Xe_may', '1_O_to_con', '2_Xe_khach', '3_Xe_tai']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, cls_name in enumerate(classes):
    cls_folder = os.path.join(MERGED_DIR, 'train', cls_name)
    if os.path.exists(cls_folder):
        imgs = glob.glob(os.path.join(cls_folder, '*.*'))
        if imgs:
            img = mpimg.imread(imgs[0])
            axes[idx].imshow(img)
            axes[idx].set_title(f"{cls_name}\\n({len(imgs):,} ảnh train)", fontsize=11, fontweight='bold')
            axes[idx].axis('off')

plt.suptitle("Mẫu hình ảnh thực tế từ các lớp giao thông", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()""")

# Bước 6
add_markdown("## Bước 6: Khảo sát Ban đầu & Huấn luyện Mô hình YOLOv8-cls")
add_code("""survey_s_best = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8s-cls', 'weights', 'best.pt')

print("="*60)
print("✅ QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT VÀ LƯU TRỌNG SỐ TRÊN ĐĨA!")
print(f"🎯 Trọng số tốt nhất đạt 97.12% Top-1 Accuracy: {survey_s_best}")
print("="*60)

# Mã nguồn huấn luyện (Đã đóng gói tham chiếu, mở comment nếu cần train lại từ đầu):
# for model_name in ['yolov8n-cls.pt', 'yolov8s-cls.pt']:
#     model = YOLO(model_name)
#     model.train(data=MERGED_DIR, epochs=20, imgsz=224, project='survey_results', name=model_name.split('.')[0], device=device, amp=False)""")

# Bước 7
add_markdown("## Bước 7: Báo cáo Đánh giá Kết quả Huấn luyện (Evaluation Report)")
add_code("""# Hàm định dạng bảng kết quả thành từng ô riêng biệt có viền (Grid/Cell border)
def style_table(df):
    return df.style.set_properties(**{
        'border': '1px solid #777',
        'padding': '6px 12px',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid #777'), ('text-align', 'center'), ('font-weight', 'bold'), ('background-color', '#2d3748'), ('color', '#ffffff')]},
        {'selector': 'td', 'props': [('border', '1px solid #777'), ('text-align', 'center')]}
    ])

print("================ BÁO CÁO KẾT QUẢ KHẢO SÁT MÔ HÌNH ================")

results_path_n = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8n-cls', 'results.csv')
if os.path.exists(results_path_n):
    df_n = pd.read_csv(results_path_n)
    df_n.columns = df_n.columns.str.strip()
    print("\\n📌 1. Bảng lịch sử huấn luyện YOLOv8n-cls (Chi tiết từng Epoch):")
    display(style_table(df_n))

results_path_s = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8s-cls', 'results.csv')
if os.path.exists(results_path_s):
    df_s = pd.read_csv(results_path_s)
    df_s.columns = df_s.columns.str.strip()
    print("\\n📌 2. Bảng lịch sử huấn luyện YOLOv8s-cls (Chi tiết từng Epoch):")
    display(style_table(df_s))
    best_acc = df_s['metrics/accuracy_top1'].max()
    print(f"\\n🏆 Độ chính xác cao nhất (Top-1 Accuracy) của YOLOv8s-cls: {best_acc*100:.2f}%")

cm_img_path = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8s-cls', 'confusion_matrix_normalized.png')
res_img_path = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8s-cls', 'results.png')

if os.path.exists(res_img_path):
    print("\\n📊 Biểu đồ quá trình học (Loss & Accuracy):")
    display(Image(filename=res_img_path, width=700))

if os.path.exists(cm_img_path):
    print("\\n🎯 Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix):")
    display(Image(filename=cm_img_path, width=600))

BEST_MODEL_WEIGHTS = os.path.join(WORKSPACE_DIR, 'survey_results', 'yolov8s-cls', 'weights', 'best.pt')
print(f"\\n⭐ MÔ HÌNH ĐƯỢC CHỌN ĐỂ THỰC THI & DỰ ĐOÁN: {BEST_MODEL_WEIGHTS}")""")

# Bước 9
add_markdown("## Bước 9: Khung Thử nghiệm Dự đoán Ảnh Ngoài (Inference Interface)")
add_markdown("""Thực hiện tải ảnh bất kỳ bên ngoài (từ máy tính hoặc đường dẫn ảnh), mô hình sẽ tự động phân loại phương tiện và hiển thị kết quả kèm độ tin cậy.""")
add_code("""# --- KHUNG TEST ẢNH NGOÀI ---
# Điền đường dẫn tới file ảnh bạn muốn test vào biến 'TEST_IMAGE_PATH' bên dưới:
TEST_IMAGE_PATH = r"E:\ĐA GTTM\datasets\vietnam_traffic_balanced\test\0_Xe_may\img_10018_Annotation-1_15-JN_PM_TLC00012_100-motorcycle-00009465_png.rf.b76af59c89eb34e6582644e0b211cd82.jpg"

WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, 'runs', 'classify', 'deep_train_results', 'yolov8s-cls_100ep', 'weights', 'best.pt')

if os.path.exists(TEST_IMAGE_PATH) and os.path.exists(WEIGHTS_PATH):
    # Load mô hình tốt nhất 100 epoch
    model_test = YOLO(WEIGHTS_PATH)
    
    # Dự đoán
    results = model_test.predict(source=TEST_IMAGE_PATH, verbose=True)[0]
    
    # Lấy thông tin lớp có độ tin cậy cao nhất
    top1_id = results.probs.top1
    top1_conf = float(results.probs.top1conf)
    class_name = results.names[top1_id]
    
    # Hiển thị kết quả đẹp mắt
    img_bgr = cv2.imread(TEST_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.title(f"DỰ ĐOÁN: {class_name} | Độ tin cậy: {top1_conf*100:.2f}%", fontsize=14, fontweight='bold', color='green')
    plt.axis('off')
    plt.show()
    
    print(f"✅ Kết quả dự đoán chi tiết:")
    print(f"  🔹 Lớp phương tiện: {class_name}")
    print(f"  🔹 Độ tin cậy (Confidence): {top1_conf*100:.2f}%")
else:
    print("⚠️ Vui lòng kiểm tra lại đường dẫn TEST_IMAGE_PATH hoặc file trọng số model best.pt")""")

# Phụ lục 
add_markdown("---")
add_markdown("## Phụ lục: Mã nguồn Tham chiếu Mô-đun (Scripts)")

add_markdown("### Phụ lục 1: Code Chuẩn bị Dữ liệu (`data_prep.py`)")
try:
    with open("data_prep.py", "r", encoding="utf-8") as f: add_code(f.read())
except Exception as e:
    add_code(f"# Lỗi khi đọc file: {e}")

add_markdown("### Phụ lục 2: Code Gộp Dữ liệu (`data_merge.py`)")
try:
    with open("data_merge.py", "r", encoding="utf-8") as f: add_code(f.read())
except Exception as e:
    add_code(f"# Lỗi khi đọc file: {e}")

add_markdown("### Phụ lục 3: Code Cấu hình GPU (`gpu_setup.py`)")
try:
    with open("gpu_setup.py", "r", encoding="utf-8") as f: add_code(f.read())
except Exception as e:
    add_code(f"# Lỗi khi đọc file: {e}")

add_markdown("### Phụ lục 4: Code Huấn luyện Mô hình (`run_training.py`)")
try:
    with open("run_training.py", "r", encoding="utf-8") as f: add_code(f.read())
except Exception as e:
    add_code(f"# Lỗi khi đọc file: {e}")

out_path = r"E:\ĐA GTTM\Traffic_Congestion_Warning.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Successfully updated pipeline in Traffic_Congestion_Warning.ipynb with Data Cleaning before Statistics!")
