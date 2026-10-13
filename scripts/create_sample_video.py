import os
import glob
import cv2
import numpy as np

from config import BALANCED_DATASET_DIR, SAMPLES_DIR


def imread_unicode(path):
    """
    Đọc ảnh an toàn với đường dẫn Unicode (tiếng Việt) trên Windows.

    cv2.imread() KHÔNG hỗ trợ đường dẫn chứa ký tự không phải ASCII
    (ví dụ "E:\\ĐA GTTM\\...") và luôn trả về None.
    Cách đúng: np.fromfile đọc bytes -> cv2.imdecode giải mã ảnh.
    """
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def create_sample_video():
    test_dir = os.path.join(str(BALANCED_DATASET_DIR), "test")
    dest_path = os.path.join(str(SAMPLES_DIR), "sample_traffic.mp4")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Collect images from test dataset
    img_files = []
    for cls in ["0_Xe_may", "1_O_to_con", "2_Xe_khach", "3_Xe_tai"]:
        cls_dir = os.path.join(test_dir, cls)
        if os.path.exists(cls_dir):
            files = glob.glob(os.path.join(cls_dir, "*.*"))
            img_files.extend(files[:15])

    if not img_files:
        print("[ERROR] No image files found to build sample video.")
        return

    # Target resolution
    target_w, target_h = 854, 480
    fps = 10

    # VideoWriter using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dest_path, fourcc, fps, (target_w, target_h))

    frame_count = 0
    skipped = 0
    for img_path in img_files:
        img = imread_unicode(img_path)
        if img is None:
            skipped += 1
            print(f"[WARNING] Không đọc được ảnh (bỏ qua): {os.path.basename(img_path)}")
            continue
        resized = cv2.resize(img, (target_w, target_h))
        
        # Write each image frame multiple times to simulate video stream with transitions
        for _ in range(8):
            out.write(resized)
            frame_count += 1

    out.release()

    size_kb = os.path.getsize(dest_path) / 1024 if os.path.exists(dest_path) else 0
    print(f"[SUCCESS] Đã tạo video mẫu: {dest_path}")
    print(f"          {frame_count} frame | {size_kb:.1f} KB | bỏ qua {skipped} ảnh lỗi")

    if frame_count == 0:
        print("[ERROR] Không ghi được frame nào - video sẽ hỏng!")
    else:
        # Xác minh video đọc lại được
        probe = cv2.VideoCapture(dest_path)
        ok = probe.isOpened()
        count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        probe.release()
        print(f"[VERIFY] Mở video thành công: {ok} | Số frame đọc được: {count}")

if __name__ == "__main__":
    create_sample_video()
