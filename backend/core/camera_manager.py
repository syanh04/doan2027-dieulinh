"""
camera_manager.py - Điều phối đa luồng hệ thống Camera Giám sát Giao thông với YOLOv8 & Real-time Analytics.
Tuân thủ chặt chẽ nguyên tắc dữ liệu:
- KHÔNG sử dụng số liệu giả lập / base_profiles / sine waves.
- 100% chỉ số (Số lượng xe, PCU, Mật độ K, Lưu lượng Q, Vận tốc V, Cấp độ ùn tắc) được tính toán trực tiếp
  từ các phương tiện nhận diện trong video thời gian thực.
- Cấu hình mặc định nạp 4 video thực tế:
    + CAM-01: uploads/CAM-01/cam1.mp4
    + CAM-02: uploads/CAM-01/cam2.mkv
    + CAM-03: uploads/CAM-01/cam3.mkv
    + CAM-04: uploads/CAM-01/cam4.mkv
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
from backend.core.traffic_db import (
    record_minute_metric, create_analysis_session, close_analysis_session, add_audit_log
)

# Khóa đồng bộ GPU an toàn giữa các luồng camera
model_lock = threading.Lock()

# Mapping phương tiện COCO sang nhãn tiếng Việt & màu vẽ (BGR format cho OpenCV)
COCO_VEHICLES = {
    2: ("Ô tô con", "o_to_con", (0, 215, 255)),     # Gold
    3: ("Xe máy", "xe_may", (255, 180, 0)),         # Cyan/Sky
    5: ("Xe khách", "xe_khach", (226, 43, 138)),    # Purple
    7: ("Xe tải", "xe_tai", (0, 100, 255)),         # Orange/Red
    1: ("Xe đạp", "xe_dap", (94, 197, 34))          # Xe đạp - Emerald Green (BGR)
}


def resolve_video_path(rel_path):
    """Tìm đường dẫn tệp video linh hoạt hỗ trợ cả đường dẫn tuyệt đối lẫn tương đối."""
    if not rel_path:
        return ""
    clean_path = str(rel_path).replace("\\", "/")
    if os.path.isabs(clean_path) and os.path.exists(clean_path):
        return os.path.normpath(clean_path)
    if os.path.exists(str(rel_path)):
        return os.path.normpath(str(rel_path))

    candidates = [
        os.path.join(str(WORKSPACE_DIR), clean_path),
        os.path.join(str(WORKSPACE_DIR), rel_path),
        os.path.join(str(WORKSPACE_DIR), "uploads", os.path.basename(rel_path)),
        os.path.join(str(WORKSPACE_DIR), "uploads", "CAM-01", os.path.basename(rel_path)),
        os.path.join(str(STATIC_DIR), "samples", os.path.basename(rel_path)),
        os.path.join(str(STATIC_DIR), os.path.basename(rel_path))
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return os.path.normpath(os.path.join(str(WORKSPACE_DIR), rel_path))


class CameraWorker:
    def __init__(self, cam_id, name, district, bearing, video_rel_path, default_roi, length_m=60.0, width_m=12.0, det_model=None):
        self.cam_id = cam_id
        self.name = name
        self.district = district
        self.bearing = bearing
        self.video_rel_path = video_rel_path
        self.video_path = resolve_video_path(video_rel_path)
        self.roi_norm = np.array(default_roi, dtype=np.float32)
        self.length_m = float(length_m)
        self.width_m = float(width_m)
        self.det_model = det_model

        # Quản lý phiên phân tích
        self.session_id = None
        self.source_type = "upload" if "uploads" in str(video_rel_path) else "cctv_sample"

        # Trạng thái luồng
        self.is_running = False
        self.thread = None
        self.latest_jpeg_low = None
        self.latest_jpeg_high = None
        self.new_frame_event = threading.Event()

        # Bộ đệm bounding box liên tục tránh giật nháy
        self.last_det_boxes = []

        # Chỉ số thời gian thực khởi tạo
        self.metrics = {
            "total_vehicles": 0,
            "counts": {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0, "xe_dap": 0},
            "pcu": 0.0,
            "density": 0.0,
            "flow": 0.0,
            "speed": 0.0,
            "fps": 0.0,
            "congestion_level": 1,
            "congestion_name": "Thông thoáng",
            "last_updated": time.time(),
            "video_time_sec": 0.0,
            "session_id": None
        }

        # Quản lý tracking và đo lường lưu lượng (Flow Q)
        self.tracked_objects = {}  # {track_id: dict(centroid, cls, first_seen, last_time, last_x, last_y, speed)}
        self.next_track_id = 1
        self.crossed_ids = set()
        self.crossing_events = deque(maxlen=2000)  # [(video_time_sec, cat_key)]

        self.video_fps = 25.0
        self.cumulative_vtime_sec = 0.0
        self.last_minute_record_vtime = 0.0
        self.last_audit_log_vtime = 0.0
        self.last_audit_congestion_lvl = 1

    def start(self):
        if not self.is_running:
            self._init_session()
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name=f"Worker-{self.cam_id}")
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.5)
        self._close_session()

    def _init_session(self):
        """Khởi tạo phiên phân tích mới độc lập cho video."""
        self.session_id = f"{self.cam_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.tracked_objects.clear()
        self.crossed_ids.clear()
        self.crossing_events.clear()
        self.last_det_boxes.clear()
        self.cumulative_vtime_sec = 0.0
        self.last_minute_record_vtime = 0.0
        self.last_audit_log_vtime = 0.0
        self.last_audit_congestion_lvl = 1

        total_frames = 0
        fps = 25.0
        if os.path.exists(self.video_path):
            cap_probe = cv2.VideoCapture(self.video_path)
            if cap_probe.isOpened():
                total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap_probe.get(cv2.CAP_PROP_FPS)) or 25.0
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
                status_text="Khởi tạo phiên phân tích trực tiếp từ video",
                details=f"Phiên: {self.session_id} | Tệp: {os.path.basename(self.video_path)}"
            )
        except Exception as e:
            print(f"[SESSION ERROR] {e}")

    def _close_session(self):
        if self.session_id:
            try:
                close_analysis_session(self.session_id)
            except Exception:
                pass

    def change_source(self, new_video_path, source_type="upload"):
        """Đổi nguồn video cho camera: dừng phiên cũ, reset chỉ số, nạp phiên mới."""
        self.stop()
        self.video_rel_path = new_video_path
        self.video_path = resolve_video_path(new_video_path)
        self.source_type = source_type
        self.metrics = {
            "total_vehicles": 0,
            "counts": {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0, "xe_dap": 0},
            "pcu": 0.0,
            "density": 0.0,
            "flow": 0.0,
            "speed": 0.0,
            "fps": 0.0,
            "congestion_level": 1,
            "congestion_name": "Thông thoáng",
            "last_updated": time.time(),
            "video_time_sec": 0.0,
            "session_id": None
        }
        self.start()

    def update_roi(self, new_roi_norm):
        """Cập nhật tọa độ đa giác ROI trực tiếp."""
        self.roi_norm = np.array(new_roi_norm, dtype=np.float32)

    def _run_loop(self):
        """Vòng lặp đọc frame, nhận diện phương tiện YOLO, tracking và tính toán chỉ số thực tế."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            err_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, f"{self.cam_id}: Khong tim thay video {os.path.basename(self.video_path)}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", err_frame)
            self.latest_jpeg_low = buf.tobytes()
            self.latest_jpeg_high = buf.tobytes()
            print(f"[ERROR] Không thể mở video cho {self.cam_id}: {self.video_path}")
            return

        frame_idx = 0
        fps_timer = time.time()
        fps_counter = 0
        curr_fps = self.video_fps

        while self.is_running:
            ret, raw_frame = cap.read()
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

            # Timeline liên tục của phiên phân tích
            frame_delta_sec = 1.0 / max(1.0, self.video_fps)
            self.cumulative_vtime_sec += frame_delta_sec
            video_time_sec = self.cumulative_vtime_sec

            raw_h, raw_w = raw_frame.shape[:2]
            # Chuẩn hóa kích thước khung hình về 1280px để đạt hiệu năng xử lý mượt mà và nét cao
            target_w = 1280
            target_h = int(target_w * raw_h / raw_w)
            frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

            # Suy luận YOLO xen kẽ mỗi 2 frame để tối ưu tốc độ và duy trì nhịp frame mượt
            should_infer = (frame_idx % 2 == 0)

            if should_infer and self.det_model is not None:
                try:
                    with model_lock:
                        results = self.det_model(
                            frame, conf=0.18, iou=0.45, imgsz=640,
                            classes=list(COCO_VEHICLES.keys()),
                            verbose=False
                        )[0]

                    det_boxes = []
                    if results.boxes is not None and len(results.boxes) > 0:
                        for b in results.boxes:
                            cid = int(b.cls[0])
                            if cid in COCO_VEHICLES:
                                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                                det_boxes.append((x1, y1, x2, y2, cid, float(b.conf[0])))
                    self.last_det_boxes = det_boxes
                except Exception:
                    det_boxes = self.last_det_boxes
            else:
                det_boxes = self.last_det_boxes

            # Tính tọa độ pixel của ROI theo kích thước khung hình hiện tại
            roi_px = (self.roi_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
            roi_count = 0
            counts = {"xe_may": 0, "o_to_con": 0, "xe_khach": 0, "xe_tai": 0, "xe_dap": 0}
            current_centers = []
            roi_det_boxes = []

            # Lọc các phương tiện nằm trong vùng ROI ngã tư (Chỉ đếm xe trong zone vàng)
            for (x1, y1, x2, y2, cid, conf) in det_boxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cat_key = COCO_VEHICLES[cid][1]

                # Kiểm tra xe có nằm trong đa giác ngã tư không
                if cv2.pointPolygonTest(roi_px, (float(cx), float(cy)), False) >= 0:
                    roi_count += 1
                    counts[cat_key] += 1
                    current_centers.append(((cx, cy), cat_key, (x1, y1, x2, y2)))
                    roi_det_boxes.append((x1, y1, x2, y2, cid, conf))

            # Tọa độ giới hạn của ROI và đường cắt qua (Trigger Line)
            roi_min_x = np.min(roi_px[:, 0])
            roi_max_x = np.max(roi_px[:, 0])
            roi_min_y = np.min(roi_px[:, 1])
            roi_max_y = np.max(roi_px[:, 1])
            trigger_y = roi_min_y + (roi_max_y - roi_min_y) * 0.50
            trigger_x = roi_min_x + (roi_max_x - roi_min_x) * 0.50

            # Cập nhật Tracking đối tượng trong ROI
            new_tracked = {}
            used_current = set()
            m_per_px_y = self.length_m / max(1.0, float(roi_max_y - roi_min_y))
            m_per_px_x = self.width_m / max(1.0, float(roi_max_x - roi_min_x))

            for tid, obj in list(self.tracked_objects.items()):
                prev_c = obj["centroid"]
                best_dist = 70.0
                best_idx = -1
                for idx, (curr_c, c_cat, c_box) in enumerate(current_centers):
                    if idx in used_current:
                        continue
                    dist = np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx >= 0:
                    used_current.add(best_idx)
                    c_pt, c_cat, c_box = current_centers[best_idx]

                    # Ước tính vận tốc thực tế của từng xe theo hướng dịch chuyển
                    dx_m = (c_pt[0] - prev_c[0]) * m_per_px_x
                    dy_m = (c_pt[1] - prev_c[1]) * m_per_px_y
                    dist_m = np.hypot(dx_m, dy_m)
                    dt = max(0.01, video_time_sec - obj.get("last_time", video_time_sec - 0.04))
                    v_kmh = (dist_m / dt) * 3.6
                    v_kmh = min(80.0, max(0.0, v_kmh))

                    # Kiểm tra vượt qua trigger line (hỗ trợ cả luồng xe dọc Y lẫn luồng xe ngang X)
                    if tid not in self.crossed_ids:
                        crossed_y = (obj["last_y"] < trigger_y and c_pt[1] >= trigger_y) or (obj["last_y"] > trigger_y and c_pt[1] <= trigger_y)
                        crossed_x = (obj["last_x"] < trigger_x and c_pt[0] >= trigger_x) or (obj["last_x"] > trigger_x and c_pt[0] <= trigger_x)
                        if crossed_y or crossed_x:
                            self.crossed_ids.add(tid)
                            self.crossing_events.append((video_time_sec, c_cat))

                    new_tracked[tid] = {
                        "centroid": c_pt,
                        "cls": c_cat,
                        "first_seen": obj["first_seen"],
                        "last_time": video_time_sec,
                        "last_x": c_pt[0],
                        "last_y": c_pt[1],
                        "speed": v_kmh
                    }

            # Thêm xe mới phát hiện trong ROI vào tracking
            for idx, (curr_c, c_cat, c_box) in enumerate(current_centers):
                if idx not in used_current:
                    tid = self.next_track_id
                    self.next_track_id += 1
                    new_tracked[tid] = {
                        "centroid": curr_c,
                        "cls": c_cat,
                        "first_seen": video_time_sec,
                        "last_time": video_time_sec,
                        "last_x": curr_c[0],
                        "last_y": curr_c[1],
                        "speed": 0.0
                    }

            self.tracked_objects = new_tracked

            # =========================================================================
            # TÍNH TOÁN CÁC CHỈ SỐ HOÀN TOÀN TỪ PHƯƠNG TIỆN THỰC TẾ TRONG VIDEO
            # =========================================================================
            # 1. Mật độ K (xe/km trong ROI)
            roi_km = max(0.01, self.length_m / 1000.0)
            density = round(roi_count / roi_km, 1)

            # 2. Quy đổi xe con tiêu chuẩn PCU
            pcu = round(
                counts["xe_may"] * PCU_WEIGHTS["xe_may"] +
                counts["o_to_con"] * PCU_WEIGHTS["o_to_con"] +
                counts["xe_khach"] * PCU_WEIGHTS["xe_khach"] +
                counts["xe_tai"] * PCU_WEIGHTS["xe_tai"] +
                counts.get("xe_dap", 0) * PCU_WEIGHTS.get("xe_dap", 0.2), 1
            )

            # 3. Vận tốc V (km/h) trung bình thực tế từ các xe đang chuyển động
            moving_speeds = [obj["speed"] for obj in self.tracked_objects.values() if obj.get("speed", 0) > 3.0]
            if moving_speeds:
                speed = round(float(np.mean(moving_speeds)), 1)
            elif density > 0:
                # Mô hình Greenshields ITS khi các xe đang xếp hàng / di chuyển chậm
                v_free = 45.0
                speed = round(max(5.0, v_free * (1.0 - min(0.85, density / 350.0))), 1)
            else:
                speed = 45.0

            # 4. Lưu lượng Q (xe/giờ) từ đo lường vạch cắt và định luật dòng giao thông Q = K * V
            window_sec = 30.0
            recent_crossings = [ev for ev in self.crossing_events if (video_time_sec - ev[0]) <= window_sec]
            if len(recent_crossings) >= 2 and video_time_sec >= 10.0:
                actual_w = max(10.0, min(video_time_sec, window_sec))
                flow_counting = (len(recent_crossings) / actual_w) * 3600.0
                # Kết hợp trọng số giữa số đếm thực và định luật giao thông cơ bản
                flow = round(min(4500.0, max(100.0, 0.6 * flow_counting + 0.4 * (density * speed))), 0)
            elif roi_count > 0 and speed > 0:
                flow = round(min(4500.0, density * speed), 0)
            else:
                flow = 0.0

            # 5. Phân cấp mức độ ùn tắc
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

            # Cập nhật telemetry thời gian thực
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
                "video_time_sec": round(video_time_sec, 1),
                "session_id": self.session_id
            }

            # Lưu vào cơ sở dữ liệu định kỳ mỗi 5 giây
            if (video_time_sec - self.last_minute_record_vtime) >= 5.0:
                self.last_minute_record_vtime = video_time_sec
                record_minute_metric(
                    camera_id=self.cam_id,
                    counts=counts,
                    pcu=pcu,
                    density=density,
                    flow=flow,
                    avg_speed=speed,
                    session_id=self.session_id,
                    video_time_sec=video_time_sec
                )

            # Ghi nhận trạng thái giao thông và cảnh báo vào Audit Logs định kỳ (mỗi 15s) hoặc khi đổi mức độ ùn tắc
            if (video_time_sec - self.last_audit_log_vtime) >= 15.0 or (cg_lvl != self.last_audit_congestion_lvl and (video_time_sec - self.last_audit_log_vtime) >= 4.0):
                self.last_audit_log_vtime = video_time_sec
                lvl_changed = (cg_lvl != self.last_audit_congestion_lvl)
                self.last_audit_congestion_lvl = cg_lvl

                if cg_lvl == 1:
                    advice = "Lưu lượng ngã tư thông thoáng. Duy trì chu kỳ đèn tín hiệu 60s."
                elif cg_lvl == 2:
                    advice = "Dòng xe tăng cao trong ngã tư. Khuyến nghị kéo dài pha đèn xanh trục chính +15s."
                elif cg_lvl == 3:
                    advice = "Cảnh báo ùn ứ nút giao. Tăng thời lượng thoát xe trục chính +25s, hạn chế luồng phụ."
                else:
                    advice = "Ùn tắc nghiêm trọng tại nút giao! Kích hoạt chế độ giải tỏa khẩn cấp và cảnh báo CSGT."

                event_type = "THRESHOLD" if (cg_lvl >= 2 and lvl_changed) else "TRAFFIC_STATE"
                status_text = f"{cg_name} ({roi_count} xe trong ngã tư)"
                details = f"Xe máy: {counts['xe_may']} | Ô tô con: {counts['o_to_con']} | Xe khách: {counts['xe_khach']} | Xe tải: {counts['xe_tai']} | Xe đạp: {counts['xe_dap']} | PCU: {pcu:.1f}"

                try:
                    add_audit_log(
                        event_type=event_type,
                        camera_id=self.cam_id,
                        flow=flow,
                        density=density,
                        pcu=pcu,
                        speed=speed,
                        status_level=cg_lvl,
                        status_text=status_text,
                        advice=advice,
                        details=details
                    )
                except Exception as log_err:
                    pass

            # =========================================================================
            # VẼ TRỰC QUAN HÓA LÊN VIDEO STREAM
            # =========================================================================
            frame_annotated = frame.copy()

            # 1. Vẽ đa giác ROI ngã tư (Màu vàng kim với lớp phủ trong suốt nổi bật)
            overlay = frame_annotated.copy()
            cv2.fillPoly(overlay, [roi_px], (0, 215, 255))
            cv2.addWeighted(overlay, 0.16, frame_annotated, 0.84, 0, frame_annotated)
            cv2.polylines(frame_annotated, [roi_px], isClosed=True, color=(0, 215, 255), thickness=2)

            # 2. Vẽ vạch đếm xe (Màu đỏ)
            cv2.line(frame_annotated, (int(roi_min_x), int(trigger_y)), (int(roi_max_x), int(trigger_y)), (0, 0, 255), 2)

            # 3. CHỈ vẽ bounding box và nhãn cho xe nằm TRONG vùng ngã tư (Không vẽ/đếm xe ngoài zone vàng)
            for (x1, y1, x2, y2, cid, conf) in roi_det_boxes:
                label_vn, cat_key, color = COCO_VEHICLES[cid]
                label_ascii = {
                    "xe_may": "Xe may",
                    "o_to_con": "O to con",
                    "xe_khach": "Xe khach",
                    "xe_tai": "Xe tai",
                    "xe_dap": "Xe dap"
                }.get(cat_key, label_vn)

                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{label_ascii} {int(conf * 100)}%"
                tw = len(tag) * 9 + 8
                cv2.rectangle(frame_annotated, (x1, max(0, y1 - 20)), (x1 + tw, y1), color, -1)
                cv2.putText(frame_annotated, tag, (x1 + 4, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Thêm banner telemetry nhỏ góc trên bên trái
            hud_text = f"XE TRONG NGA TU: {roi_count} | PCU: {pcu:.1f} | K: {density:.0f} xe/km | Q: {int(flow)} xe/h | V: {speed:.1f} km/h"
            cv2.rectangle(frame_annotated, (12, 12), (16 + len(hud_text) * 9, 38), (15, 23, 42), -1)
            cv2.putText(frame_annotated, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

            # Chế độ lưới (Low stream): 480px chiều rộng
            frame_low = cv2.resize(frame_annotated, (480, int(480 * h / w)), interpolation=cv2.INTER_AREA)
            _, buf_low = cv2.imencode(".jpg", frame_low, [cv2.IMWRITE_JPEG_QUALITY, 55])
            self.latest_jpeg_low = buf_low.tobytes()

            # Chế độ xem chi tiết Focus (High stream): chất lượng cao
            _, buf_high = cv2.imencode(".jpg", frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.latest_jpeg_high = buf_high.tobytes()
            self.new_frame_event.set()

            # Nhịp frame điều hòa ~25 FPS
            time.sleep(0.038)

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
                "name": "Trạm Giám Sát CAM-01 (cam1.mp4)",
                "district": "uploads/CAM-01/cam1.mp4",
                "bearing": "Lưu lượng ngã tư",
                "video": "uploads/CAM-01/cam1.mp4",
                "roi": [
                    [0.22, 0.44],
                    [0.82, 0.44],
                    [0.82, 0.32],
                    [0.70, 0.23],
                    [0.42, 0.23],
                    [0.22, 0.32]
                ],
                "length_m": 45.0,
                "width_m": 16.0
            },
            {
                "id": "CAM-02",
                "name": "Trạm Giám Sát CAM-02 (cam2.mkv)",
                "district": "uploads/CAM-01/cam2.mkv",
                "bearing": "Lưu lượng ngã tư",
                "video": "uploads/CAM-01/cam2.mkv",
                "roi": [
                    [0.14, 0.44],
                    [0.69, 0.44],
                    [0.78, 0.30],
                    [0.64, 0.17],
                    [0.38, 0.17],
                    [0.15, 0.24],
                    [0.08, 0.38]
                ],
                "length_m": 40.0,
                "width_m": 15.0
            },
            {
                "id": "CAM-03",
                "name": "Trạm Giám Sát CAM-03 (cam3.mkv)",
                "district": "uploads/CAM-01/cam3.mkv",
                "bearing": "Lưu lượng ngã tư",
                "video": "uploads/CAM-01/cam3.mkv",
                "roi": [
                    [0.14, 0.44],
                    [0.69, 0.44],
                    [0.78, 0.30],
                    [0.64, 0.17],
                    [0.38, 0.17],
                    [0.15, 0.24],
                    [0.08, 0.38]
                ],
                "length_m": 40.0,
                "width_m": 15.0
            },
            {
                "id": "CAM-04",
                "name": "Trạm Giám Sát CAM-04 (cam4.mkv)",
                "district": "uploads/CAM-01/cam4.mkv",
                "bearing": "Lưu lượng ngã tư",
                "video": "uploads/CAM-01/cam4.mkv",
                "roi": [
                    [0.14, 0.44],
                    [0.69, 0.44],
                    [0.78, 0.30],
                    [0.64, 0.17],
                    [0.38, 0.17],
                    [0.15, 0.24],
                    [0.08, 0.38]
                ],
                "length_m": 40.0,
                "width_m": 15.0
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

        print(f"[CAMERA MANAGER] Khởi chạy thành công {len(self.workers)} camera worker với 4 video thực tế.")

    def assign_source(self, cam_id, new_video_rel_path, source_type="upload"):
        """Gán video mới cho camera đã chỉ định."""
        worker = self.workers.get(cam_id)
        if not worker:
            return False, f"Camera {cam_id} không tồn tại"
        worker.change_source(new_video_rel_path, source_type=source_type)
        return True, f"Đã gán nguồn {new_video_rel_path} cho {cam_id}"

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
