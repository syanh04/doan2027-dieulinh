"""
test_video.py - Kiem thu day du luong video:
  1. Upload video qua API  (kiem tra validation + luu file)
  2. Stream MJPEG tu /video_feed/<ten>
  3. Poll /api/video_metrics de theo doi chi so giao thong theo thoi gian
"""
import io
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:5000"
VIDEO = "static/samples/traffic_scene_test.mp4"
DURATION = 90          # giay thu stream


def post_multipart(path, files):
    boundary = "----WebKitFormBoundaryTEST1234"
    body = io.BytesIO()
    for k, fname in files.items():
        with open(fname, "rb") as fh:
            data = fh.read()
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode()
        )
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        BASE + path, data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


print("=" * 70)
print("BUOC 1: Upload video qua API")
st, data = post_multipart("/api/upload_video", {"file": VIDEO})
print(f"  status={st} success={data.get('success')}")
if not data.get("success"):
    print(f"  error={data.get('error')}")
    raise SystemExit(1)
filename = data["filename"]
print(f"  filename={filename}  size_mb={data.get('size_mb')}")

print("=" * 70)
print("BUOC 2: Bat dau stream + poll chi so")
stream_stats = {"frames": 0, "bytes": 0, "error": None}


def stream_worker():
    try:
        resp = urllib.request.urlopen(
            BASE + f"/video_feed/{filename}?conf=0.25", timeout=300
        )
        while True:
            chunk = resp.read(16384)
            if not chunk:
                break
            stream_stats["frames"] += chunk.count(b"--frame\r\n")
            stream_stats["bytes"] += len(chunk)
        resp.close()
    except Exception as e:
        stream_stats["error"] = str(e)


thread = threading.Thread(target=stream_worker, daemon=True)
thread.start()

print(f"{'Giay':>6} | {'Xe':>4} | {'Xe may':>7} | {'O to':>5} | {'Khach':>6} | "
      f"{'Tai':>4} | {'PCU':>6} | {'Mdo':>7} | {'Chiem%':>7} | {'Luu luong':>9} | Muc do")
print("-" * 118)

start = time.time()
last_level = None
while time.time() - start < DURATION:
    time.sleep(1.0)
    elapsed = int(time.time() - start)
    try:
        with urllib.request.urlopen(
            BASE + f"/api/video_metrics/{filename}", timeout=10
        ) as r:
            m = json.loads(r.read().decode())["metrics"]
    except Exception:
        continue

    c = m["counts"]
    cg = m["congestion"]
    marker = ""
    if cg["level"] != last_level:
        marker = "  <== DOI MUC"
        last_level = cg["level"]

    print(f"{elapsed:>6} | {m['total_vehicles']:>4} | {c['xe_may']:>7} | "
          f"{c['o_to_con']:>5} | {c['xe_khach']:>6} | {c['xe_tai']:>4} | "
          f"{m['pcu_score']:>6} | {m['density']:>7} | {m['occupancy']:>7} | "
          f"{m['flow']:>9} | L{cg['level']} {cg['name']}{marker}")

    if m.get("stream_ended"):
        print(f"\n  [THONG BAO] Stream da dung: {m.get('stream_message')}")
        break

print("=" * 70)
print("BUOC 3: Ket qua stream")
print(f"  Frame nhan duoc : {stream_stats['frames']}")
print(f"  Kich thuoc      : {stream_stats['bytes'] / 1024:.0f} KB")
print(f"  FPS trung binh  : {stream_stats['frames'] / DURATION:.1f} frame/gia")
if stream_stats["error"]:
    print(f"  Loi            : {stream_stats['error']}")
print("=" * 70)
print("HOAN TAT KIEM THU VIDEO")