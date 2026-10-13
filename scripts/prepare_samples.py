import os
import shutil
import glob

from config import BALANCED_DATASET_DIR, SAMPLES_DIR

def prepare_samples():
    test_dir = os.path.join(str(BALANCED_DATASET_DIR), "test")
    dest_dir = str(SAMPLES_DIR)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    classes = {
        "0_Xe_may": "sample_xe_may.jpg",
        "1_O_to_con": "sample_o_to.jpg",
        "2_Xe_khach": "sample_xe_khach.jpg",
        "3_Xe_tai": "sample_xe_tai.jpg"
    }
    
    for cls_name, sample_filename in classes.items():
        cls_folder = os.path.join(test_dir, cls_name)
        if os.path.exists(cls_folder):
            files = glob.glob(os.path.join(cls_folder, "*.*"))
            if files:
                dst = os.path.join(dest_dir, sample_filename)
                shutil.copy2(files[0], dst)
                print("[SUCCESS] Copied sample image for " + cls_name)

if __name__ == "__main__":
    prepare_samples()
