import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from config import SURVEY_RESULTS_DIR

def analyze_time(csv_path):
    if not os.path.exists(csv_path):
        print(f"File không tồn tại: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    if 'time' not in df.columns:
        print("Không tìm thấy cột 'time' trong file csv.")
        return
        
    if 'epoch' not in df.columns:
        print("Không tìm thấy cột 'epoch' trong file csv.")
        return

    # Calculate time per epoch
    df['time_per_epoch'] = df['time'].diff()
    df.loc[0, 'time_per_epoch'] = df.loc[0, 'time']

    total_time_seconds = df['time'].iloc[-1]
    avg_time_seconds = df['time_per_epoch'].mean()
    
    total_hours = total_time_seconds / 3600
    avg_minutes = avg_time_seconds / 60
    
    print("="*50)
    print(" BÁO CÁO THỐNG KÊ THỜI GIAN TRAINING")
    print("="*50)
    print(f"Số lượng Epoch đã train: {len(df)}")
    print(f"Tổng thời gian (Total Time): {total_time_seconds:.2f} giây (~ {total_hours:.2f} giờ)")
    print(f"Thời gian trung bình mỗi Epoch: {avg_time_seconds:.2f} giây (~ {avg_minutes:.2f} phút)")
    print("="*50)
    
    # Print table of epochs
    print(df[['epoch', 'time', 'time_per_epoch']].to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['time_per_epoch'], marker='o', linestyle='-', color='b')
    plt.title('Thời gian tiêu tốn cho mỗi Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Thời gian (giây)')
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(csv_path), "time_analysis.png")
    plt.savefig(plot_path)
    print(f"\nĐã lưu biểu đồ thời gian tại: {plot_path}")

if __name__ == "__main__":
    # If a path is provided in arguments, use it
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default path (tự động suy ra từ config.py)
        path = os.path.join(str(SURVEY_RESULTS_DIR), "yolov8n_cls", "results.csv")
        
    analyze_time(path)
