import os
from ultralytics import YOLO
import torch

from config import WORKSPACE_DIR, BALANCED_DATASET_DIR

def main():
    DATA_DIR = os.path.join(str(BALANCED_DATASET_DIR))
    os.chdir(str(WORKSPACE_DIR))
    
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print(f"GPU IS AVAILABLE: {torch.cuda.get_device_name(0)} (device={device})")
    else:
        device = 'cpu'
        print("GPU NOT AVAILABLE! Running on CPU.")
        
    print("Working Directory:", os.getcwd())
    print("Data Directory:", DATA_DIR)

    BASE_WEIGHTS = "yolov8s-cls.pt"
    
    print(f"\n========== ĐANG TRAIN CHUYÊN SÂU MÔ HÌNH YOLOV8S-CLS TRÊN DATASET CÂN BẰNG (100 EPOCHS) ==========")
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=DATA_DIR, 
        epochs=100, 
        imgsz=224, 
        project='deep_train_results', 
        name='yolov8s-cls_100ep',
        device=device,
        amp=False,
        optimizer='AdamW',
        exist_ok=True
    )
    
    print("Quá trình huấn luyện đã hoàn tất!")
    
    print("Quá trình huấn luyện đã hoàn tất!")

if __name__ == "__main__":
    main()

