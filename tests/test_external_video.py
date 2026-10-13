"""
test_external_video.py - Thử nghiệm mô hình nhận diện và phân tích ùn tắc trên VIDEO NGOÀI.

Cách dùng:
    python test_external_video.py
    (Sau đó nhập đường dẫn video của bạn, hoặc nhấn Enter để dùng video mẫu có sẵn)
"""
import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

from config import DET_WEIGHTS_PATH, RUNS_DIR, WORKSPACE_DIR

# Mapping COCO vehicle class IDs sang tiếng Việt
COCO_VEHICLE_MAP = {
    2: ("Ô tô con", "o_to_con", (0, 215, 255)),     # Yellow/Gold
    3: ("Xe máy", "xe_may", (255, 144, 30)),        # Cyan/Blue
    5: ("Xe khách", "xe_khach", (226, 43, 138)),    # Purple
    7: ("Xe tải", "xe_tai", (0, 69, 255)),          # Orange/Red
    1: ("Xe đạp", "xe_may", (255, 144, 30))
}

PCU_WEIGHTS = {"xe_may": 0.5, "o_to_con": 1.0, "xe_khach": 2.5, "xe_tai": 2.0}

# Vùng ROI chuẩn hoá (0..1)
ROI_NORM = np.array([
    [0.12, 0.99],   # dưới-trái
    [0.88, 0.99],   # dưới-phải
    [0.66, 0.40],   # trên-phải
    [0.34, 0.40],   # trên-trái
], dtype=np.float32)

ROI_REAL_LENGTH_M = 60.0
DENSITY_KM_FACTOR = ROI_REAL_LENGTH_M / 1000.0


def get_roi_pixels(w, h):
    pts = ROI_NORM.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.int32)


def point_in_roi(roi_px, x, y):
    return cv2.pointPolygonTest(roi_px, (float(x), float(y)), False) >= 0


def analyze_congestion(roi_count, density, occupancy):
    if roi_count <= 0:
        return {"name": "Thông thoáng (Không xe)", "level": 0, "color": (150, 150, 150)}

    lvl_density = 1 if density < 120 else (2 if density < 220 else (3 if density < 350 else 4))
    lvl_occ = 1 if occupancy < 15 else (2 if occupancy < 30 else (3 if occupancy < 50 else 4))
    level = max(lvl_density, lvl_occ)

    levels = {
        1: ("Thông thoáng", (0, 200, 0)),
        2: ("Đông đúc", (0, 215, 255)),
        3: ("Ùn tắc", (0, 0, 255)),
        4: ("Ùn tắc nghiêm trọng", (0, 0, 180))
    }
    name, color = levels.get(level, ("Thông thoáng", (0, 200, 0)))
    return {"name": name, "level": level, "color": color}


def process_video(video_path, output_path=None, conf_thresh=0.25, show_window=True):
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy file video: {video_path}")
        return

    if not os.path.exists(DET_WEIGHTS_PATH):
        print(f"❌ Không tìm thấy trọng số YOLO detection tại: {DET_WEIGHTS_PATH}")
        return

    print(f"\n[INFO] Nạp mô hình YOLOv8 từ: {DET_WEIGHTS_PATH}")
    model = YOLO(str(DET_WEIGHTS_PATH))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở file video: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        os.makedirs(os.path.join(str(WORKSPACE_DIR), "runs", "predict_video"), exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(str(WORKSPACE_DIR), "runs", "predict_video", f"{base_name}_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    roi_px = get_roi_pixels(w, h)
    roi_area = max(1.0, float(cv2.contourArea(roi_px)))

    win_name = "TrafficVision AI - Window Test Video Preview"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 850, 480)

    print("=" * 65)
    print(f"🎬 ĐANG XỬ LÝ VIDEO NGOÀI: {video_path}")
    print(f"📐 Kích thước: {w}x{h} | FPS: {fps:.1f} | Tổng số frame: {total_frames}")
    print(f"🖥️  Hiển thị cửa sổ xem thử nhỏ: CÓ (bấm 'q' trên cửa sổ để dừng)")
    print(f"💾 Video kết quả sẽ lưu tại: {output_path}")
    print("=" * 65)

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated = frame.copy()

            # Chạy YOLO detection
            results = model.predict(source=frame, conf=conf_thresh, imgsz=1024, verbose=False)[0]

            counts = {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0}
            roi_count = 0
            roi_pcu = 0.0
            occupied_area = 0.0

            if results.boxes is not None:
                for b in results.boxes:
                    cid = int(b.cls[0])
                    conf = float(b.conf[0])
                    if cid not in COCO_VEHICLE_MAP:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    label_vn, cat_key, color = COCO_VEHICLE_MAP[cid]

                    # Vẽ bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    tag = f"{label_vn} {conf*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 8, y1), color, -1)
                    cv2.putText(annotated, tag, (x1 + 4, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Kiểm tra vị trí tiếp đất trong ROI
                    cx, cy = (x1 + x2) / 2.0, float(y2)
                    if point_in_roi(roi_px, cx, cy):
                        counts[cat_key] += 1
                        roi_count += 1
                        roi_pcu += PCU_WEIGHTS[cat_key]

                        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                        inter, _ = cv2.intersectConvexConvex(rect, roi_px.astype(np.float32))
                        occupied_area += max(0.0, inter)

            occupancy = (occupied_area / roi_area) * 100.0
            density = roi_count / DENSITY_KM_FACTOR
            cg = analyze_congestion(roi_count, density, occupancy)

            # Vẽ overlay ROI
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [roi_px], cg["color"])
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.polylines(annotated, [roi_px], True, cg["color"], 2)

            # HUD Thông tin
            hud_txt = f"Frame {frame_idx}/{total_frames} | Xe trong ROI: {roi_count} | Mdt: {density:.0f} xe/km | Level: {cg['name'].upper()}"
            cv2.rectangle(annotated, (10, 10), (w - 10, 50), (20, 20, 30), -1)
            cv2.putText(annotated, hud_txt, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)

            writer.write(annotated)

            if show_window:
                cv2.imshow(win_name, annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27): # 'q' hoặc ESC
                    print("\n[INFO] Đã hủy cửa sổ xem thử bởi người dùng.")
                    break

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / max(0.001, elapsed)
                print(f"  Frame {frame_idx:>4}/{total_frames} | {fps_proc:.1f} FPS | ROI: {roi_count} xe | Mật độ: {density:.0f} xe/km | Status: {cg['name']}")

    finally:
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    print("=" * 65)
    print("✅ XỬ LÝ VIDEO HOÀN TẤT!")
    print(f"⏱️ Tổng thời gian: {total_time:.2f} giây ({frame_idx/max(0.1, total_time):.1f} FPS)")
    print(f"📁 Video đã tạo: {output_path} ({size_mb:.2f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    v_path = None

    if len(sys.argv) > 1:
        v_path = sys.argv[1].strip('"')
    else:
        # Thử mở hộp thoại chọn file của Windows
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw() # Ẩn cửa sổ chính của tkinter
            root.attributes('-topmost', True) # Đưa hộp thoại lên trên cùng
            selected = filedialog.askopenfilename(
                title="Chọn file video ngoài để thử nghiệm TrafficVision AI",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")]
            )
            root.destroy()
            if selected:
                v_path = selected
        except Exception:
            pass

    if not v_path:
        default_sample = os.path.join(str(WORKSPACE_DIR), "static", "samples", "sample_traffic.mp4")
        print("💡 Nhập đường dẫn file video ngoài (hoặc nhấn ENTER để dùng video mẫu sẵn có):")
        inp = input("👉 Đường dẫn video: ").strip('"')
        v_path = inp if inp else default_sample

    process_video(v_path, show_window=True)

