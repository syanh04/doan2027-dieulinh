import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
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

add_markdown("# Hệ thống Nhận diện và Phân loại Phương tiện Giao Thông (Cảnh Báo Ùn Tắc)\nBài toán: Phân loại các loại xe lưu thông trên đường (Xe máy, Ô tô con, Xe khách, Xe tải) từ ảnh cắt camera.\nPhương pháp: Image Classification với YOLOv8-cls.")

add_markdown("## 1. Kết nối Thư viện")
add_code("import os\nimport glob\nimport random\nimport cv2\nimport matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\nimport ultralytics\nfrom ultralytics import YOLO\nimport torch\n\nprint(\"Ultralytics version:\", ultralytics.__version__)\nprint(\"PyTorch version:\", torch.__version__)")

add_markdown("## 2. Kiểm tra GPU")
add_code("if torch.cuda.is_available():\n    print(\"GPU IS AVAILABLE:\", torch.cuda.get_device_name(0))\nelse:\n    print(\"GPU NOT AVAILABLE! Môi trường đang chạy trên CPU.\")")

add_markdown("## 3. Cấu hình Dữ liệu (Đã chuẩn bị sẵn bằng Script)\nDữ liệu đã được gộp từ 2 dataset và đưa về định dạng thư mục Classification: `vietnam_traffic_cls`.")
add_code("WORKSPACE_DIR = r\"D:\\ĐA GTTM\"\nDATA_DIR = os.path.join(WORKSPACE_DIR, \"datasets\", \"vietnam_traffic_cls\")\nos.chdir(WORKSPACE_DIR)\nprint(\"Working Directory:\", os.getcwd())\nprint(\"Data Directory:\", DATA_DIR)")

add_markdown("## 4. Kiểm tra Nhanh Data sau Làm Sạch\nHiển thị thử 1 bức ảnh từ tập train.")
add_code("train_folders = glob.glob(os.path.join(DATA_DIR, \"train\", \"*\"))\nif train_folders:\n    sample_class = train_folders[0]\n    images = glob.glob(os.path.join(sample_class, \"*.jpg\"))\n    if images:\n        img = mpimg.imread(images[0])\n        plt.imshow(img)\n        plt.axis('off')\n        plt.title(f'Ảnh mẫu: {os.path.basename(sample_class)}')\n        plt.show()")

add_markdown("## 5. Khảo sát 3 mô hình trong 20 Epoch\nChúng ta sẽ train thử YOLOv8n-cls, YOLOv8s-cls trong 20 epoch để chọn mô hình tốt nhất.")
add_code("epochs_survey = 20\nmodels_to_test = ['yolov8n-cls.pt', 'yolov8s-cls.pt']\n\n# Bỏ comment đoạn sau để bắt đầu khảo sát:\n# for model_name in models_to_test:\n#     print(f\"\\n========== ĐANG TRAIN MÔ HÌNH {model_name} ==========\")\n#     model = YOLO(model_name)\n#     model.train(data=DATA_DIR, epochs=epochs_survey, imgsz=224, project='survey_results', name=model_name.split('.')[0])\nprint(\"Code khảo sát đã sẵn sàng.\")")

add_markdown("## 6. Đánh giá và Chọn mô hình tốt nhất")
add_code("import pandas as pd\nimport os\nfrom IPython.display import display\n\nresults_path_n = os.path.join('survey_results', 'yolov8n-cls', 'results.csv')\nif os.path.exists(results_path_n):\n    df_n = pd.read_csv(results_path_n)\n    df_n.columns = df_n.columns.str.strip()\n    print(\"\\n--- BẢNG THỐNG KÊ MÔ HÌNH YOLOv8n-cls ---\")\n    display(df_n.tail())\n\nresults_path_s = os.path.join('survey_results', 'yolov8s-cls', 'results.csv')\nif os.path.exists(results_path_s):\n    df_s = pd.read_csv(results_path_s)\n    df_s.columns = df_s.columns.str.strip()\n    print(\"\\n--- BẢNG THỐNG KÊ MÔ HÌNH YOLOv8s-cls ---\")\n    display(df_s.tail())\n\nBEST_MODEL_WEIGHTS = \"survey_results/yolov8s-cls/weights/best.pt\"\nprint(f\"\\nMô hình được chọn để train chuyên sâu: {BEST_MODEL_WEIGHTS}\")")

add_markdown("## 7. Chạy Chuyên Sâu 100 Epoch với Mô hình tốt nhất")
add_code("# Bỏ comment đoạn sau để train chuyên sâu 100 epoch:\n# best_model = YOLO(BEST_MODEL_WEIGHTS)\n# best_model.train(data=DATA_DIR, epochs=100, imgsz=224, project='deep_train_results', name='best_model_100ep', resume=True)\nprint(\"Code train chuyên sâu 100 epoch đã sẵn sàng.\")")

with open(r"D:\ĐA GTTM\Traffic_Congestion_Warning.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
print("Updated Traffic_Congestion_Warning.ipynb")
