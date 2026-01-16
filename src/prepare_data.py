import os
import shutil
import random
import sys

# 🔥 加入這行，確保可以從根目錄讀取到 src 資料夾
sys.path.append(os.getcwd())

try:
    from src.config import RAW_BREAD_MAP as NAME_MAP
except ImportError:
    print("錯誤: 找不到 src/config.py，請確認檔案結構。")
    exit()

PROJECT_ROOT = "D:\\Project\\ML\\Bread_classifier"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test")

TEST_RATIO = 0.1         
VAL_RATIO_OF_REST = 0.2  

# 清空資料夾內容
def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def prepare_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"錯誤: 找不到原始資料夾: {RAW_DATA_PATH}")
        print(f"當前路徑: {PROJECT_ROOT}")
        return

    print("=" * 50)
    print("開始執行 3 階段資料切分 (Train / Val / Test)")
    print("=" * 50)

    print("清理舊資料...", end="")
    clear_directory(TRAIN_DATA_PATH)
    clear_directory(VAL_DATA_PATH)
    clear_directory(TEST_DATA_PATH)
    print("完成！\n")

    chinese_folders = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]

    total_images_count = 0
    processed_classes = 0

    for chinese_name in chinese_folders:
        if chinese_name not in NAME_MAP:
            print(f"跳過: 資料夾 '{chinese_name}' 未在設定檔中")
            continue

        english_name = NAME_MAP[chinese_name]
        processed_classes += 1
        
        # 建立資料夾結構
        for path in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]:
            os.makedirs(os.path.join(path, english_name), exist_ok=True)

        class_dir = os.path.join(RAW_DATA_PATH, chinese_name)
        all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not all_files:
            continue

        random.shuffle(all_files)
        
        n_total = len(all_files)
        total_images_count += n_total

        n_test = int(n_total * TEST_RATIO)
        rest_files = all_files[n_test:]
        n_rest = len(rest_files)
        n_val = int(n_rest * VAL_RATIO_OF_REST)
        
        test_files = all_files[:n_test]
        val_files = rest_files[:n_val]
        train_files = rest_files[n_val:]

        for f in test_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(TEST_DATA_PATH, english_name, f))
        for f in val_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(VAL_DATA_PATH, english_name, f))
        for f in train_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(TRAIN_DATA_PATH, english_name, f))

        print(f"{english_name:<25} | 總數: {n_total:<3} -> Train: {len(train_files):<3} | Val: {len(val_files):<3} | Test: {len(test_files):<3}")

    print("\n" + "=" * 50)
    print(f"處理完成！共 {processed_classes} 個類別，{total_images_count} 張圖片。")
    print("=" * 50)

if __name__ == "__main__":
    prepare_data()