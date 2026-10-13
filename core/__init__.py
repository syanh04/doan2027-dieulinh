"""
core package - TrafficVision Enterprise Core Domain Modules.
Bao gom:
- camera_manager: Dieu phoi da luong worker camera, tracking ByteTrack & YOLOv8
- forecaster: Mo hinh du bao chuoi thoi gian (Holt's Damped Trend, MA, Backtest)
- traffic_db: He quan tri SQLite, auth nguong, phan quyen va chuoi so lieu
- alerts: He thong phat hien va canh bao un tac giao thong
"""
from core.traffic_db import *
from core.forecaster import forecaster, TrafficForecaster
from core.camera_manager import camera_manager, CameraManager
