import os
import shutil
import random

# --- 設定變數 ---
PROJECT_ROOT = "bread_classifier"
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test")

# 切分比例設定
TEST_RATIO = 0.1         # 10% 作為測試集 (Test)
VAL_RATIO_OF_REST = 0.2  # 剩下的 90% 中，取 20% 作為驗證集 (Val) 其餘為訓練集 (Train)

# 名稱對應表 (中文 -> 英文)
NAME_MAP = {
    "丹麥咖啡奶酥波羅": "danish_coffee_polo",
    "丹麥奶酥波蘿": "danish_butter_polo",
    "丹麥波蘿": "danish_polo",
    "核桃布里奧斯": "walnut_brioche",
    "白麵包-巧克力": "white_bread_choc",
    "白麵包-草莓": "white_bread_straw",
    "肉鬆沙拉": "rousong_salad",
    "花生麵包": "peanut_bread",
    "起酥肉鬆": "puff_rousong",
    "香蒜巴西里": "garlic_parsley",
}

#清空資料夾內容
def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def prepare_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"錯誤: 找不到原始資料夾: {RAW_DATA_PATH}")
        return

    print("=" * 50)
    print("開始執行 3 階段資料切分 (Train / Val / Test)")
    print(f"目標: 測試集 {TEST_RATIO*100}% | 剩餘部分再分 驗證 {VAL_RATIO_OF_REST*100}% / 訓練 80%")
    print("=" * 50)

    print("清理舊資料...", end="")
    clear_directory(TRAIN_DATA_PATH)
    clear_directory(VAL_DATA_PATH)
    clear_directory(TEST_DATA_PATH)
    print("完成！\n")

    #取得原有資料集
    chinese_folders = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]

    total_images_count = 0

    for chinese_name in chinese_folders:
        if chinese_name not in NAME_MAP:
            print(f"該資料夾未處於對應表中: {chinese_name}")
            continue

        english_name = NAME_MAP[chinese_name]
        
        # 建立子資料夾
        os.makedirs(os.path.join(TRAIN_DATA_PATH, english_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DATA_PATH, english_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DATA_PATH, english_name), exist_ok=True)

        # 讀取並打亂檔案
        class_dir = os.path.join(RAW_DATA_PATH, chinese_name)
        all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_files)
        
        n_total = len(all_files)
        total_images_count += n_total

        # --- 計算切分數量 ---
        #測試集 Test
        n_test = int(n_total * TEST_RATIO)
        
        #驗證集 Val & 訓練集 Train (除去測試集後的剩餘部分)
        rest_files = all_files[n_test:]
        n_rest = len(rest_files)
        
        #驗證集 Val
        n_val = int(n_rest * VAL_RATIO_OF_REST)
        
        #訓練集 Train 
        test_files = all_files[:n_test]
        val_files = rest_files[:n_val]
        train_files = rest_files[n_val:]

        #移動檔案
        for f in test_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(TEST_DATA_PATH, english_name, f))
        
        for f in val_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(VAL_DATA_PATH, english_name, f))
            
        for f in train_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(TRAIN_DATA_PATH, english_name, f))

        print(f"{english_name:<20} | 總數: {n_total:<3} -> Train: {len(train_files):<3} | Val: {len(val_files):<3} | Test: {len(test_files):<3}")

    print("\n" + "=" * 50)
    print("資料集準備完成！")
    print(f"訓練集位置 (Train): {TRAIN_DATA_PATH}")
    print(f"驗證集位置 (Val)  : {VAL_DATA_PATH}")
    print(f"測試集位置 (Test) : {TEST_DATA_PATH}")
    print("=" * 50)

if __name__ == "__main__":
    prepare_data()