"""
Nhan dien xe + canh bao un tac cho 1 doan duong (video file).

Pipeline:
  YOLO + ByteTrack -> loc xe trong ROI -> doi pixel sang met (homography)
  -> tinh mat do + toc do trung binh -> trung binh truot theo thoi gian
  -> may trang thai (hysteresis + thoi gian duy tri) -> canh bao
"""
import math
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# Lấy kích thước ROI/ngưỡng dùng chung từ config.py (cùng 1 nguồn với app.py).
try:
    from config import ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M, DENSITY_THRESHOLDS
except Exception:
    ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M = 10.0, 40.0
    DENSITY_THRESHOLDS = [120.0, 220.0, 350.0]

# ======================= CAU HINH (SUA THEO DOAN DUONG CUA BAN) =======================
VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"     # dat None neu khong muon luu video
SHOW = True                    # hien cua so xem truc tiep (nhan 'q' de thoat)

MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
CONF = 0.3

# 4 diem ROI tren anh (pixel), THU TU: duoi-trai, duoi-phai, tren-phai, tren-trai
# (duoi = gan camera, tren = xa camera). Hay thay bang toa do that cua ban.
ROI_PTS = np.array([[200, 700], [1000, 700], [800, 300], [400, 300]], dtype=np.float32)

# Kich thuoc THUC cua vung ROI (met) lay tu config.py (cung 1 nguon voi app.py).
# Do rong mat duong va chieu dai doan duong dung de doi pixel sang met & tinh xe/km.
# De hieu chuan, dat ROI_REAL_LENGTH_M / ROI_REAL_WIDTH_M trong .env (xem .env.example).

# Trung binh truot
WINDOW_SEC = 30.0        # cua so lam muot so lieu
SPEED_TRACK_SEC = 1.0    # do dai lich su de tinh toc do moi xe
MIN_TRACK_POINTS = 5     # so diem toi thieu de tinh toc do 1 xe
MAX_SPEED_KMH = 120.0    # loai nhieu

# Nguong canh bao (HAY HIEU CHINH THEO DU LIEU THUC)
DENSITY_ON = 80.0        # xe/km: vuot nguong nay (va toc do thap) -> co the un
DENSITY_OFF = 50.0       # xe/km: xuong duoi nguong nay -> co the het un
SPEED_ON = 20.0          # km/h: thap hon -> co the un
SPEED_OFF = 30.0         # km/h: cao hon -> co the het un
HOLD_ON_SEC = 20.0       # dieu kien un phai keo dai lien tuc bay lau moi bao
HOLD_OFF_SEC = 15.0      # dieu kien het un phai keo dai bay lau moi tat bao
# =======================================================================================


def build_homography():
    """Anh -> mat phang duong (met). Goc toa do: duoi-trai cua ROI, Y huong ra xa camera."""
    dst = np.array(
        [[0, 0], [ROI_REAL_WIDTH_M, 0],
         [ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M], [0, ROI_REAL_LENGTH_M]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(ROI_PTS, dst)


def to_meters(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def in_roi(x, y):
    return cv2.pointPolygonTest(ROI_PTS.astype(np.int32), (float(x), float(y)), False) >= 0


class CongestionMonitor:
    """Trung binh truot + may trang thai co hysteresis va thoi gian duy tri."""

    def __init__(self):
        self.window = deque()      # (t, so_xe, toc_do_hoac_nan)
        self.congested = False
        self.cond_since = None     # thoi diem dieu kien chuyen trang thai bat dau dung

    def update(self, t, n, speed):
        self.window.append((t, n, speed))
        while self.window and t - self.window[0][0] > WINDOW_SEC:
            self.window.popleft()

        counts = [w[1] for w in self.window]
        speeds = [w[2] for w in self.window if not math.isnan(w[2])]
        n_avg = float(np.mean(counts))
        v_avg = float(np.mean(speeds)) if speeds else None
        density = n_avg / (ROI_REAL_LENGTH_M / 1000.0)  # xe/km

        if not self.congested:
            cond = density >= DENSITY_ON and v_avg is not None and v_avg <= SPEED_ON
            hold = HOLD_ON_SEC
        else:
            cond = density <= DENSITY_OFF or (v_avg is not None and v_avg >= SPEED_OFF)
            hold = HOLD_OFF_SEC

        changed = False
        if cond:
            if self.cond_since is None:
                self.cond_since = t
            if t - self.cond_since >= hold:
                self.congested = not self.congested
                self.cond_since = None
                changed = True
        else:
            self.cond_since = None

        return density, v_avg, changed


def main():
    model = YOLO(MODEL_PATH)
    H = build_homography()
    monitor = CongestionMonitor()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if OUTPUT_PATH:
        writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history = {}     # track_id -> deque[(t, X, Y)] (met)
    last_seen = {}   # track_id -> t
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps  # dung thoi gian video, khong phu thuoc toc do xu ly
        frame_idx += 1

        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          classes=VEHICLE_CLASSES, conf=CONF, verbose=False)[0]

        n_in_roi = 0
        speeds = []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx, cy = (x1 + x2) / 2, y2  # diem tiep dat: giua canh duoi bbox
                if not in_roi(cx, cy):
                    continue
                n_in_roi += 1

                X, Y = to_meters(H, cx, cy)
                dq = history.setdefault(int(tid), deque())
                dq.append((t, X, Y))
                last_seen[int(tid)] = t
                while dq and t - dq[0][0] > SPEED_TRACK_SEC:
                    dq.popleft()

                color = (0, 255, 0)
                if len(dq) >= MIN_TRACK_POINTS and dq[-1][0] - dq[0][0] > 0.3:
                    dist = math.hypot(dq[-1][1] - dq[0][1], dq[-1][2] - dq[0][2])
                    v = dist / (dq[-1][0] - dq[0][0]) * 3.6
                    if v <= MAX_SPEED_KMH:
                        speeds.append(v)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # don dep track cu
        for tid in [k for k, v in last_seen.items() if t - v > 2.0]:
            history.pop(tid, None)
            last_seen.pop(tid, None)

        v_frame = float(np.mean(speeds)) if speeds else float("nan")
        density, v_avg, changed = monitor.update(t, n_in_roi, v_frame)

        if changed:
            state = "BAT DAU UN TAC" if monitor.congested else "HET UN TAC"
            print(f"[{t:7.1f}s] {state} | mat do={density:.0f} xe/km | "
                  f"toc do TB={v_avg if v_avg is not None else float('nan'):.1f} km/h")

        # ------------------ ve overlay ------------------
        color = (0, 0, 255) if monitor.congested else (0, 200, 0)
        cv2.polylines(frame, [ROI_PTS.astype(np.int32)], True, color, 2)
        label = "CONGESTED" if monitor.congested else "FREE FLOW"
        v_txt = f"{v_avg:.1f}" if v_avg is not None else "--"
        cv2.rectangle(frame, (0, 0), (430, 95), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Vehicles in ROI: {n_in_roi}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Density: {density:.0f} veh/km  Speed: {v_txt} km/h",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if writer:
            writer.write(frame)
        if SHOW:
            cv2.imshow("Congestion", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
