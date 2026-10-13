import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from config import CLS_WEIGHTS_PATH


def imread_unicode(path):
    """
    Đọc ảnh an toàn với đường dẫn Unicode (tiếng Việt) trên Windows.
    cv2.imread() trả về None nếu đường dẫn chứa ký tự không phải ASCII.
    """
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def predict_external_image(image_path):
    WEIGHTS_PATH = str(CLS_WEIGHTS_PATH)

    if not os.path.exists(WEIGHTS_PATH):
        print("❌ Không tìm thấy file trọng số mô hình best.pt tại:", WEIGHTS_PATH)
        return

    if not os.path.exists(image_path):
        print("❌ File ảnh nhập vào không tồn tại:", image_path)
        return

    # Load mô hình YOLOv8s-cls tốt nhất
    model = YOLO(WEIGHTS_PATH)

    # Dự đoán
    results = model.predict(source=image_path, verbose=False)[0]
    
    top1_id = results.probs.top1
    top1_conf = float(results.probs.top1conf)
    class_name = results.names[top1_id]

    print("\n" + "="*50)
    print("🚗 KẾT QUẢ DỰ ĐOÁN NHẬN DẠNG PHƯƠNG TIỆN 🚗")
    print(f"👉 Đường dẫn ảnh: {image_path}")
    print(f"👉 Phương tiện nhận diện: {class_name}")
    print(f"👉 Độ tin cậy (Confidence): {top1_conf * 100:.2f}%")
    print("="*50)

    # Đọc và hiển thị ảnh
    img_bgr = imread_unicode(image_path)
    if img_bgr is None:
        print("❌ Không đọc được ảnh (đường dẫn sai hoặc định dạng không hỗ trợ):", image_path)
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)
    plt.title(f"DỰ ĐOÁN: {class_name} ({top1_conf*100:.2f}%)", fontsize=14, fontweight='bold', color='green')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Điền đường dẫn ảnh ngoài của bạn vào đây:
    test_img = input("Nhập đường dẫn file ảnh bạn muốn test: ").strip('"')
    if test_img:
        predict_external_image(test_img)
