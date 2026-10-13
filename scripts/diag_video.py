"""
diag_video.py - Chan doan vi sao khong phat hien xe trong video test.
So sanh: detection tho (bo qua ROI) vs detection nam trong ROI.
"""
import os
import sys

import cv2
import numpy as np

from ultralytics import YOLO
from config import DET_WEIGHTS_PATH

VIDEO = "static/samples/traffic_scene_test.mp4"
OUT_DIR = "diag"

COCO_VEHICLE_MAP = {
    2: ("O to con", (0, 215, 255)),
    3: ("Xe may", (255, 144, 30)),
    5: ("Xe khach", (226, 43, 138)),
    7: ("Xe tai", (0, 69, 255)),
}

# ROI chuan hoa cua app.py
ROI_NORM = np.array([
    [0.12, 0.99], [0.88, 0.99], [0.66, 0.40], [0.34, 0.40],
], dtype=np.float32)


def point_in_roi(roi_px, x, y):
    return cv2.pointPolygonTest(roi_px, (float(x), float(y)), False) >= 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(str(DET_WEIGHTS_PATH))
    cap = cv2.VideoCapture(VIDEO)

    # Lay cac frame tai cac moc thoi gian khac nhau (dau / giua / cuoi)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = [5, 60, 150, 240, 295]
    print(f"Video: {total} frame")

    for idx in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  frame {idx}: doc that bai")
            continue

        h, w = frame.shape[:2]
        roi_px = (ROI_NORM * np.array([w, h], dtype=np.float32)).astype(np.int32)

        for conf in (0.25, 0.05):
            res = model.predict(source=frame, conf=conf, imgsz=1024, verbose=False)[0]
            raw = 0
            in_roi = 0
            confs = []
            labels = []
            for b in res.boxes:
                cid = int(b.cls[0])
                if cid not in COCO_VEHICLE_MAP:
                    continue
                raw += 1
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                cx, cy = (x1 + x2) / 2.0, float(y2)
                confs.append(round(float(b.conf[0]), 2))
                labels.append(COCO_VEHICLE_MAP[cid][0])
                if point_in_roi(roi_px, cx, cy):
                    in_roi += 1

            print(f"  frame {idx:>3} conf={conf}: thoi={raw:>2} | trongROI={in_roi:>2} "
                  f"| confs={sorted(confs, reverse=True)[:8]}")
            if conf == 0.05 and labels:
                print(f"           nhan: {labels[:8]}")

        # Luu anh kiem tra
        cv2.polylines(frame, [roi_px], True, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx}.jpg"), frame)

    cap.release()
    print(f"\nAnh kiem tra luu tai: {OUT_DIR}/")


if __name__ == "__main__":
    main()