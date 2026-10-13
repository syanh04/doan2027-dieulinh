import torch
import ultralytics

def get_gpu_config():
    """
    Tự động kiểm tra và trả về cấu hình GPU tối ưu nhất cho PyTorch và YOLO.
    Giúp chống lỗi FP16 (amp=False) và luôn cố định chạy trên GPU 0.
    """
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        print(f"[GPU SETUP] DA NHAN GPU thanh cong: {device_name}")
        print(f"[GPU SETUP] CUDA Version: {torch.version.cuda} | PyTorch: {torch.__version__}")
        device = 0
    else:
        print("[GPU SETUP] KHONG TIM THAY GPU! Dang dung CPU.")
        device = 'cpu'
        
    config = {
        'device': device,
        'amp': False if cuda_ok else True,  # amp=False giup tranh loi FP16 CUBLAS tren RTX Windows
        'workers': 4 if cuda_ok else 2
    }
    return device, config

if __name__ == "__main__":
    device, config = get_gpu_config()
    print("[TEST RESULT] GPU Config ready:", config)
