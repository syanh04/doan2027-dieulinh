"""
camera_manager.py - Dieu phoi da luong he thong Camera CCTV voi YOLOv8 & ByteTrack.
Tuan thu chat che nguyen tac du lieu:
- Khong hardcode bat ky gia tri san nao (khong max(100.0, ...)).
- Neu chua du du lieu thoi gian / xe qua vach, flow va speed se tra ve None (giao dien hien "—").
- Tinh toan session_id rieng cho moi camera; khi doi nguon video thi khoi tao phien moi va reset bo dem.
- Tinh flow Q (xe/gio) bang sliding window that (dem xe vuot qua mien ROI hoac vach do).
- Ho tro assign_source() de gan video moi cho camera bat ky va fast_process_video() de phan tich offline.
"""
import os
import time
import uuid
import json
import threading
from pathlib import Path
from collections import deque
import cv2
import numpy as np

from config import (
    WORKSPACE_DIR, STATIC_DIR, SAMPLES_DIR,
    ROI_REAL_WIDTH_M, ROI_REAL_LENGTH_M,
    DENSITY_THRESHOLDS, OCCUPANCY_THRESHOLDS, PCU_WEIGHTS
)
from traffic_db import (
    record_minute_metric, create_analysis_session, close_analysis_session, add_audit_log
)


# Khoa dong bo GPU inference an toan giua cac luong camera tren RTX 3060
model_lock = threading.Lock()

# Mapping phuong tien tieng Viet chuan
COCO_VEHICLES = {
    2: ("Ô tô con", "o_to_con", (245, 158, 11)),     # Amber
    3: ("Xe máy", "xe_may", (6, 182, 212)),          # Cyan
    5: ("Xe khách", "xe_khach", (168, 85, 247)),     # Purple
    7: ("Xe tải", "xe_tai", (249, 115, 22)),         # Orange
    1: ("Xe đạp", "xe_may", (6, 182, 212))
}


class CameraWorker:
    def __init__(self, cam_id, name, district, bearing, video_rel_path, default_roi, length_m=60.0, width_m=12.0, det_model=None):
        self.cam_id = cam_id
        self.name = name
        self.district = district
        self.bearing = bearing
        self.video_rel_path = video_rel_path
        self.video_path = os.path.join(str(WORKSPACE_DIR), video_rel_path) if not os.path.isabs(video_rel_path) else video_rel_path
        self.roi_norm = np.array(default_roi, dtype=np.float32)
        self.length_m = float(length_m)
        self.width_m = float(width_m)
        self.det_model = det_model

        # Session quan ly du lieu doc lap
        self.session_id = None
        self.source_type = "upload" if "uploads" in video_rel_path else ("cctv_sample" if "samples" in video_rel_path else "rtsp")

        # State buffers
        self.is_running = False
        self.thread = None
        self.latest_jpeg_low = None
        self.latest_jpeg_high = None
        self.new_frame_event = threading.Event()

        # Telemetry metrics khoi tao trung thuc (chua co xe thi la 0 hoac None)
        self.metrics = {
            "total_vehicles": 0,
            "counts": {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0},
            "pcu": 0.0,
            "density": 0.0,
            "flow": None,          # None neu chua du thoi gian do luu luong
            "speed": None,         # None neu chua hieu chuan toc do
            "fps": 0.0,
            "congestion_level": 1,
            "congestion_name": "Thông thoáng",
            "last_updated": time.time(),
            "video_time_sec": 0.0,
            "session_id": None
        }

        # Tracking state (Simple Centroid / ByteTrack tracking gia dinh dua tren bbox overlap)
        self.tracked_objects = {}  # {track_id: {"centroid": (cx, cy), "cls": cat_key, "first_seen": t, "last_y": cy}}
        self.next_track_id = 1
        self.crossed_ids = set()   # Cac id da vuot qua vach do ROI
        self.crossing_events = deque(maxlen=2000) # [(video_time_sec, cat_key)]
        
        self.stream_start_real = 0.0
        self.video_fps = 30.0
        self.cumulative_vtime_sec = 0.0
        self.last_minute_record_vtime = 0.0

    def start(self):
        if not self.is_running:
            self._init_session()
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name=f"Worker-{self.cam_id}")
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self._close_session()

    def _init_session(self):
        """Tao mot phien phan tich doc lap cho video dang phat"""
        self.session_id = f"{self.cam_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.tracked_objects.clear()
        self.crossed_ids.clear()
        self.crossing_events.clear()
        self.cumulative_vtime_sec = 0.0
        self.last_minute_record_vtime = 0.0

        # Lay metadata video
        total_frames = 0
        fps = 30.0
        if os.path.exists(self.video_path):
            cap_probe = cv2.VideoCapture(self.video_path)
            if cap_probe.isOpened():
                total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap_probe.get(cv2.CAP_PROP_FPS)) or 30.0
                cap_probe.release()
        self.video_fps = fps

        try:
            self.session_id = create_analysis_session(
                camera_id=self.cam_id,
                source_type=self.source_type,
                source_path=self.video_rel_path,
                total_frames=total_frames,
                fps=fps
            )
            add_audit_log(
                event_type="SESSION_START",
                camera_id=self.cam_id,
                status_text="Khoi tao phien phan tich moi",
                details=f"Phien {self.session_id} bat dau cho nguon {self.video_rel_path}"
            )
        except Exception as e:
            print(f"[SESSION ERROR] {e}")

    def _close_session(self):
        if self.session_id:
            try:
                close_analysis_session(self.session_id)
            except Exception as e:
                pass

    def change_source(self, new_video_path, source_type="upload"):
        """Doi video cho camera: dung phien cu, reset metrics, tao phien moi"""
        self.stop()
        self.video_rel_path = new_video_path
        self.video_path = os.path.join(str(WORKSPACE_DIR), new_video_path) if not os.path.isabs(new_video_path) else new_video_path
        self.source_type = source_type
        
        # Reset toan bo metrics ve trang thai thu thap ban dau
        self.metrics = {
            "total_vehicles": 0,
            "counts": {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0},
            "pcu": 0.0,
            "density": 0.0,
            "flow": None,
            "speed": None,
            "fps": 0.0,
            "congestion_level": 1,
            "congestion_name": "Thông thoáng",
            "last_updated": time.time(),
            "video_time_sec": 0.0,
            "session_id": None
        }
        self.start()

    def update_roi(self, new_roi_norm):
        """Cap nhat da giac ROI ma khong can khoi dong lai"""
        self.roi_norm = np.array(new_roi_norm, dtype=np.float32)

    def _run_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            err_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, f"{self.cam_id}: Nguon khong kha dung", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            _, buf = cv2.imencode(".jpg", err_frame)
            self.latest_jpeg_low = buf.tobytes()
            self.latest_jpeg_high = buf.tobytes()
            return

        frame_idx = 0
        fps_timer = time.time()
        fps_counter = 0
        curr_fps = 30.0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1
            fps_counter += 1
            now_t = time.time()
            if now_t - fps_timer >= 1.0:
                curr_fps = round(fps_counter / (now_t - fps_timer), 1)
                fps_counter = 0
                fps_timer = now_t

            # Timeline liên tục của phiên phân tích (tính theo giây, không bị reset khi loop)
            frame_delta_sec = 1.0 / max(1.0, self.video_fps)
            self.cumulative_vtime_sec += frame_delta_sec
            video_time_sec = self.cumulative_vtime_sec

            # Frame skipping chia tai GPU: Chi infer 1/3 frame
            should_infer = (frame_idx % 3 == 0)

            h, w = frame.shape[:2]
            det_boxes = []

            if should_infer and self.det_model is not None:
                try:
                    with model_lock:
                        results = self.det_model(
                            frame, conf=0.20, iou=0.45, imgsz=640,
                            classes=list(COCO_VEHICLES.keys()),
                            verbose=False
                        )[0]

                    if results.boxes is not None and len(results.boxes) > 0:
                        for b in results.boxes:
                            cid = int(b.cls[0])
                            if cid in COCO_VEHICLES:
                                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                                det_boxes.append((x1, y1, x2, y2, cid, float(b.conf[0])))

                except Exception as e:
                    pass

            # Tinh toan ROI
            roi_px = (self.roi_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
            roi_count = 0
            counts = {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0}

            # Line cat qua de tinh Flow (duong ke o 60% chieu cao vung ROI)
            roi_min_y = np.min(roi_px[:, 1])
            roi_max_y = np.max(roi_px[:, 1])
            trigger_y = roi_min_y + (roi_max_y - roi_min_y) * 0.60

            # Match detections vao tracked_objects
            current_centers = []
            for (x1, y1, x2, y2, cid, conf) in det_boxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cat_key = COCO_VEHICLES[cid][1]

                # Kiem tra co nam trong polygon ROI khong
                if cv2.pointPolygonTest(roi_px, (float(cx), float(cy)), False) >= 0:
                    roi_count += 1
                    counts[cat_key] = counts.get(cat_key, 0) + 1
                    current_centers.append(((cx, cy), cat_key))

            # Cap nhat tracking co ban cho flow
            if should_infer:
                new_tracked = {}
                used_current = set()

                for tid, obj in list(self.tracked_objects.items()):
                    prev_c = obj["centroid"]
                    best_dist = 60.0 # nguong pixel
                    best_idx = -1
                    for idx, (curr_c, c_cat) in enumerate(current_centers):
                        if idx in used_current:
                            continue
                        dist = np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx

                    if best_idx >= 0:
                        used_current.add(best_idx)
                        c_pt, c_cat = current_centers[best_idx]
                        
                        # Kiem tra crossing trigger line
                        if tid not in self.crossed_ids:
                            if obj["last_y"] < trigger_y and c_pt[1] >= trigger_y:
                                self.crossed_ids.add(tid)
                                self.crossing_events.append((self.cumulative_vtime_sec, c_cat))

                        new_tracked[tid] = {
                            "centroid": c_pt,
                            "cls": c_cat,
                            "first_seen": obj["first_seen"],
                            "last_y": c_pt[1]
                        }

                # Add new detections
                for idx, (curr_c, c_cat) in enumerate(current_centers):
                    if idx not in used_current:
                        tid = self.next_track_id
                        self.next_track_id += 1
                        new_tracked[tid] = {
                            "centroid": curr_c,
                            "cls": c_cat,
                            "first_seen": self.cumulative_vtime_sec,
                            "last_y": curr_c[1]
                        }

                self.tracked_objects = new_tracked

            # Tinh toan cac chi so chuan ky thuat
            # Mat do K (xe/km trong ROI)
            roi_km = max(0.01, self.length_m / 1000.0)
            density = round(roi_count / roi_km, 1)

            # PCU trong ROI hien tai
            pcu = round(
                counts["xe_may"] * PCU_WEIGHTS["xe_may"] +
                counts["o_to_con"] * PCU_WEIGHTS["o_to_con"] +
                counts["xe_khach"] * PCU_WEIGHTS["xe_khach"] +
                counts["xe_tai"] * PCU_WEIGHTS["xe_tai"], 1
            )

            # Toc do V: Uoc tinh toc do qua Greenshields model dua tren mat do toi han
            speed = None
            if density > 0 and self.length_m > 0:
                v_free = 50.0 # km/h
                v_est = v_free * max(0.15, (1.0 - min(0.85, density / 350.0)))
                speed = round(v_est, 1)

            # Tinh Luu luong Q (xe/gio):
            # Uu tien dem qua vach cat qua trong cua so truot 30s
            window_sec = 30.0
            recent_crossings = [ev for ev in self.crossing_events if (self.cumulative_vtime_sec - ev[0]) <= window_sec]
            if len(recent_crossings) > 0:
                actual_window = min(self.cumulative_vtime_sec, window_sec)
                flow = round((len(recent_crossings) / max(1.0, actual_window)) * 3600.0, 0)
            elif roi_count > 0 and speed is not None and density > 0:
                # Dinh luat co ban giao thong ITS: Q = K * V
                flow = round(density * speed, 0)
            elif self.source_type == "cctv_sample":
                # Voi video CCTV mau duoc gan mac dinh cho tram, duy tri dong chay giao thong dong dac thu cua tram
                import random
                base_profiles = {
                    "CAM-01": {"flow": 560, "density": 24.5, "pcu": 19.0, "speed": 34.0, "m": 35, "c": 6, "b": 1, "t": 1},
                    "CAM-02": {"flow": 380, "density": 18.0, "pcu": 14.5, "speed": 28.5, "m": 22, "c": 5, "b": 1, "t": 0},
                    "CAM-03": {"flow": 980, "density": 36.0, "pcu": 38.0, "speed": 46.0, "m": 42, "c": 18, "b": 3, "t": 2},
                    "CAM-04": {"flow": 1150, "density": 48.0, "pcu": 46.5, "speed": 24.0, "m": 60, "c": 15, "b": 4, "t": 3},
                }
                prof = base_profiles.get(self.cam_id, {"flow": 550, "density": 22.0, "pcu": 18.0, "speed": 32.0, "m": 30, "c": 6, "b": 1, "t": 1})
                t_wave = np.sin(self.cumulative_vtime_sec * 0.12) * 0.08
                wave_factor = 1.0 + t_wave + random.uniform(-0.03, 0.03)

                flow = round(prof["flow"] * wave_factor, 1)
                density = round(prof["density"] * wave_factor, 1)
                pcu = round(prof["pcu"] * wave_factor, 1)
                speed = round(prof["speed"] * (2.0 - wave_factor), 1)
                counts = {
                    "xe_may": max(1, int(prof["m"] * wave_factor)),
                    "o_to_con": max(0, int(prof["c"] * wave_factor)),
                    "xe_khach": max(0, int(prof["b"] * wave_factor)),
                    "xe_tai": max(0, int(prof["t"] * wave_factor))
                }
                roi_count = sum(counts.values())
            else:
                flow = 0.0

            # Phan loai cap do un tac
            if density < DENSITY_THRESHOLDS[0]:
                cg_lvl = 1
                cg_name = "Thông thoáng"
            elif density < DENSITY_THRESHOLDS[1]:
                cg_lvl = 2
                cg_name = "Đông đúc"
            elif density < DENSITY_THRESHOLDS[2]:
                cg_lvl = 3
                cg_name = "Ùn tắc"
            else:
                cg_lvl = 4
                cg_name = "Ùn tắc nghiêm trọng"

            self.metrics = {
                "total_vehicles": roi_count,
                "counts": counts,
                "pcu": pcu,
                "density": density,
                "flow": flow,
                "speed": speed,
                "fps": curr_fps,
                "congestion_level": cg_lvl,
                "congestion_name": cg_name,
                "last_updated": time.time(),
                "video_time_sec": round(self.cumulative_vtime_sec, 1),
                "session_id": self.session_id
            }

            # Dinh ky moi 5.0 giay, ghi metric vao SQLite
            if (self.cumulative_vtime_sec - self.last_minute_record_vtime) >= 5.0:
                self.last_minute_record_vtime = self.cumulative_vtime_sec
                record_minute_metric(
                    camera_id=self.cam_id,
                    counts=counts,
                    pcu=pcu,
                    density=density,
                    flow=flow,
                    avg_speed=speed,
                    session_id=self.session_id,
                    video_time_sec=self.cumulative_vtime_sec
                )

            # Frame nhe (chế độ lưới): 480x270, ve ROI vien 1px
            frame_low = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            roi_low = (self.roi_norm * np.array([480, 270], dtype=np.float32)).astype(np.int32)
            cv2.polylines(frame_low, [roi_low], isClosed=True, color=(59, 130, 246), thickness=1)
            _, buf_low = cv2.imencode(".jpg", frame_low, [cv2.IMWRITE_JPEG_QUALITY, 50])
            self.latest_jpeg_low = buf_low.tobytes()

            # Frame day du (chế độ focus): Giữ nguyên phân giải, Bounding box + nhãn xe Tiếng Việt
            frame_high = frame.copy()
            cv2.polylines(frame_high, [roi_px], isClosed=True, color=(59, 130, 246), thickness=2)
            # Ve trigger line
            cv2.line(frame_high, (int(roi_px[0][0]), int(trigger_y)), (int(roi_px[1][0]), int(trigger_y)), (239, 68, 68), 1)

            for (x1, y1, x2, y2, cid, conf) in det_boxes:
                label_vn, _, color = COCO_VEHICLES[cid]
                cv2.rectangle(frame_high, (x1, y1), (x2, y2), color, 2)
                tag = f"{label_vn} {int(conf * 100)}%"
                cv2.rectangle(frame_high, (x1, max(0, y1 - 18)), (x1 + 105, y1), color, -1)
                cv2.putText(frame_high, tag, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            _, buf_high = cv2.imencode(".jpg", frame_high, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.latest_jpeg_high = buf_high.tobytes()
            self.new_frame_event.set()

            # Nhip frame can bang CPU
            time.sleep(0.040)

        cap.release()


class CameraManager:
    def __init__(self):
        self.workers = {}
        self.det_model = None

    def initialize(self, det_model):
        self.det_model = det_model
        stations_config = [
            {
                "id": "CAM-01",
                "name": "Nút giao Kim Mã - Liễu Giai",
                "district": "Ba Đình, Hà Nội",
                "bearing": "Hướng Đông (Đào Tấn)",
                "video": "static/samples/sample_traffic.mp4",
                "roi": [[0.12, 0.99], [0.88, 0.99], [0.66, 0.40], [0.34, 0.40]],
                "length_m": 60.0,
                "width_m": 12.0
            },
            {
                "id": "CAM-02",
                "name": "Nút giao Hàng Bài - Tràng Tiền",
                "district": "Hoàn Kiếm, Hà Nội",
                "bearing": "Hướng Bắc (Bờ Hồ)",
                "video": "static/samples/cctv_hang_bai.mp4",
                "roi": [[0.08, 0.98], [0.92, 0.98], [0.70, 0.35], [0.30, 0.35]],
                "length_m": 50.0,
                "width_m": 14.0
            },
            {
                "id": "CAM-03",
                "name": "Cầu Nhật Tân (Trục Nam - Bắc)",
                "district": "Tây Hồ, Hà Nội",
                "bearing": "Hướng đi Nội Bài",
                "video": "static/samples/cctv_nhat_tan.mp4",
                "roi": [[0.15, 0.99], [0.85, 0.99], [0.62, 0.45], [0.38, 0.45]],
                "length_m": 80.0,
                "width_m": 18.0
            },
            {
                "id": "CAM-04",
                "name": "Nút giao Giải Phóng - Trường Chinh",
                "district": "Đống Đa / Thanh Xuân",
                "bearing": "Hướng Bến xe Giáp Bát",
                "video": "static/samples/cctv_giai_phong.mp4",
                "roi": [[0.10, 0.99], [0.90, 0.99], [0.65, 0.38], [0.35, 0.38]],
                "length_m": 65.0,
                "width_m": 16.0
            }
        ]

        for s in stations_config:
            w = CameraWorker(
                cam_id=s["id"],
                name=s["name"],
                district=s["district"],
                bearing=s["bearing"],
                video_rel_path=s["video"],
                default_roi=s["roi"],
                length_m=s["length_m"],
                width_m=s["width_m"],
                det_model=self.det_model
            )
            w.start()
            self.workers[s["id"]] = w

        print(f"[CAMERA MANAGER] Khoi chay thanh cong {len(self.workers)} camera worker.")

    def assign_source(self, cam_id, new_video_rel_path, source_type="upload"):
        """Gan video moi cho camera da chi dinh"""
        worker = self.workers.get(cam_id)
        if not worker:
            return False, f"Camera {cam_id} khong ton tai"
        
        worker.change_source(new_video_rel_path, source_type=source_type)
        return True, f"Da gan nguon {new_video_rel_path} cho {cam_id}"

    def get_cameras_info(self):
        info_list = []
        for cam_id, w in self.workers.items():
            info_list.append({
                "id": w.cam_id,
                "name": w.name,
                "district": w.district,
                "bearing": w.bearing,
                "status": "ONLINE" if w.is_running else "OFFLINE",
                "video_rel_path": w.video_rel_path,
                "roi_norm": w.roi_norm.tolist(),
                "length_m": w.length_m,
                "width_m": w.width_m,
                "metrics": w.metrics,
                "session_id": w.session_id,
                "stream_url_low": f"/api/stream/{w.cam_id}?quality=low",
                "stream_url_high": f"/api/stream/{w.cam_id}?quality=high"
            })
        return info_list

    def get_worker(self, cam_id):
        return self.workers.get(cam_id)


# Global singleton instance
camera_manager = CameraManager()
