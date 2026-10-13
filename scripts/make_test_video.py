"""
make_test_video.py - Tao video giao thong that cho chuc nang upload video.

LY DO VIET LAI (loi truoc do):
    Ban cu dan anh CAN CANH (den pha, loc tan nhiet) len nhau -> thanh so
    luon hon, khong co nao la duong pho. UI hien thi video ay -> loi "tai video".

Cach lam moi:
    1. Quet anh tu cac dataset (anh goc co toan canh duong / xe)
    2. Dung chinh mo hinh YOLO de loc: chi giu anh co >= 2 xe (day la anh
       canh giao thong that)
    3. Xep cac anh do thanh video slide (co zoom nhe cho chay)
    -> Video nhin giong duong that va YOLO van nhan duoc xe -> chi so dung.
"""
import os
import glob

import cv2
import numpy as np

from ultralytics import YOLO
from config import WORKSPACE_DIR, DATASETS_DIR, STATIC_DIR, DET_WEIGHTS_PATH

W, H = 1280, 720
FPS = 10
HOLD_FRAMES = 15         # 1.5 giay / anh
TARGET_IMAGES = 40       # so anh canh duong can lay
MIN_VEHICLES = 2         # anh la canh duong khi nhan >= 2 xe

SRC_DIRS = [
    os.path.join(str(DATASETS_DIR), "vehicles-classification-1", "train"),
    os.path.join(str(DATASETS_DIR), "Vehicle-Models-1", "train"),
    os.path.join(str(DATASETS_DIR), "vietnam_traffic_balanced", "test"),
]
OUT_DIR = os.path.join(str(STATIC_DIR), "samples")
OUT_PATH = os.path.join(OUT_DIR, "traffic_scene_test.mp4")

COCO_VEHICLES = [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck


def imread_unicode(path):
    """Doc anh an toan voi duong dan Unicode (tieng Viet) tren Windows."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"imread_unicode chi nhan duong dan, nhan duoc {type(path).__name__}")
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def collect_images(limit_per_dir=250):
    """Thu toan anh goc tu cac dataset (lay ca dataset, khong phan loai)."""
    imgs, seen = [], set()
    for d in SRC_DIRS:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "**", "*.*"), recursive=True):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            name = os.path.basename(f)
            if name in seen:
                continue
            seen.add(name)
            imgs.append(f)
            if len(imgs) >= limit_per_dir:
                return imgs
    return imgs


def filter_traffic_scenes(model, paths):
    """
    Loc anh canh giao thong: chay detection, chi giu anh co >= MIN_VEHICLES xe.
    Day la buoc quyet dinh video co dung la canh duong hay khong.
    """
    kept, scanned = [], 0
    for p in paths:
        img = imread_unicode(p)
        if img is None or img.shape[0] < 200 or img.shape[1] < 200:
            continue
        scanned += 1
        try:
            res = model.predict(source=img, conf=0.35, imgsz=640, verbose=False)[0]
        except Exception:
            continue
        n_veh = 0
        if res.boxes is not None:
            n_veh = sum(1 for c in res.boxes.cls.tolist() if int(c) in COCO_VEHICLES)
        if n_veh >= MIN_VEHICLES:
            kept.append(p)   # LUU DUONG DAN (khong luu anh da decode)
        if len(kept) >= TARGET_IMAGES:
            break
        if scanned % 50 == 0:
            print(f"  ... quet {scanned} anh, duoc {len(kept)} anh canh duong")
    return kept, scanned


def letterbox(img, tw=W, th=H):
    """
    Chuyen ti le giu nguyen hinh anh.
    DUONG BIAN DUOC THAY BANG ANH SEO MO (khong con vien den) -> nhin sach hon
    nhieu so voi cach giu vien den cu truoc day.
    """
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    resized = cv2.resize(img, (nw, nh))

    # Nen: phong to anh goc ra khung roi lam mo manh
    scale = max(tw / w, th / h)
    bg = cv2.resize(img, (int(w * scale), int(h * scale)))
    bh, bw = bg.shape[:2]
    y0b, x0b = max(0, (bh - th) // 2), max(0, (bw - tw) // 2)
    bg = bg[y0b:y0b + th, x0b:x0b + tw]
    if bg.shape[:2] != (th, tw):
        bg = cv2.resize(bg, (tw, th))
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=25)
    bg = (bg * 0.75).astype(np.uint8)

    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    bg[y0:y0 + nh, x0:x0 + nw] = resized
    return bg


def ken_burns(img, t, phase):
    """
    Zoom / translate nhe theo thoi gian de video khong trang.
    Moi anh co huong khac de khong nham chan.
    """
    s = 1.0 + 0.10 * t
    h, w = img.shape[:2]
    cw, ch = int(w / s), int(h / s)
    u = 1.0 if phase % 2 == 0 else t
    v = 1.0 if phase < 2 else t
    x0 = max(0, int((w - cw) * u))
    y0 = max(0, int((h - ch) * v))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (w, h))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1/3] Dang lay danh sach anh goc...")
    paths = collect_images()
    print(f"  -> {len(paths)} anh goc")

    print("[2/3] Dang loc anh canh duong bang YOLO (yeu cau >= 2 xe)...")
    cache_file = os.path.join(str(WORKSPACE_DIR), "_scene_cache.txt")
    scenes = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as fh:
                scenes = [ln.strip() for ln in fh if ln.strip() and os.path.exists(ln.strip())]
        except OSError:
            scenes = []
        print(f"  -> dung ket qua cache: {len(scenes)} anh")

    if len(scenes) < 3:
        model = YOLO(str(DET_WEIGHTS_PATH))
        scenes, scanned = filter_traffic_scenes(model, paths)
        print(f"  -> quet {scanned} anh, chon duoc {len(scenes)} anh canh duong")
        if len(scenes) >= 3:
            try:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(scenes))
            except OSError:
                pass

    if len(scenes) < 3:
        print("[ERROR] Khong du anh canh duong de tao video.")
        return

    print("[3/3] Dang ghep video...")
    writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frame_total = 0
    for i, path in enumerate(scenes):
        raw = imread_unicode(path)
        if raw is None:
            continue
        base = letterbox(raw)
        for f in range(HOLD_FRAMES):
            t = f / max(HOLD_FRAMES - 1, 1)
            frame = ken_burns(base, t, i % 4)
            writer.write(frame)
            frame_total += 1
        if i % 10 == 0:
            print(f"  ... da gop {i}/{len(scenes)} anh")
    writer.release()

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"[SUCCESS] {OUT_PATH}")
    print(f"          {size_mb:.2f} MB | {frame_total} frame | {frame_total/FPS:.1f}s")

    probe = cv2.VideoCapture(OUT_PATH)
    ok = probe.isOpened()
    n = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    probe.release()
    print(f"[VERIFY] Mo duoc: {ok} | Frame doc lai: {n}")


if __name__ == "__main__":
    main()