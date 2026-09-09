import os
from ultralytics import YOLO
import torch

def main():
    WORKSPACE_DIR = r"D:\ĐA GTTM"
    DATA_DIR = os.path.join(WORKSPACE_DIR, "datasets", "vietnam_traffic_cls")
    os.chdir(WORKSPACE_DIR)
    
    # Check GPU
    if torch.cuda.is_available():
        print("GPU IS AVAILABLE:", torch.cuda.get_device_name(0))
    else:
        print("GPU NOT AVAILABLE! Running on CPU.")
        
    print("Working Directory:", os.getcwd())
    print("Data Directory:", DATA_DIR)

    # Use the best weights from the 20-epoch survey
    BEST_MODEL_WEIGHTS = os.path.join("survey_results", "yolov8s-cls", "weights", "best.pt")
    
    print(f"\n========== ĐANG TRAIN CHUYÊN SÂU MÔ HÌNH (100 EPOCHS) ==========")
    print(f"Loading weights from: {BEST_MODEL_WEIGHTS}")
    
    # Load model with best weights
    best_model = YOLO(BEST_MODEL_WEIGHTS)
    
    # Start training for 100 epochs on GPU, with AMP disabled to avoid CUBLAS error
    best_model.train(
        data=DATA_DIR, 
        epochs=100, 
        imgsz=224, 
        project='deep_train_results', 
        name='best_model_100ep',
        device='0',
        amp=False,
        optimizer='AdamW'
    )
    
    print("Quá trình huấn luyện đã hoàn tất!")

if __name__ == "__main__":
    main()
