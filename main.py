import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predict import BreadPredictor

try:
    from src.get_bread_roi import get_bread_roi
except ImportError:
    print("警告: 找不到 src.get_bread_roi，將無法顯示綠框。")
    def get_bread_roi(path, text):
        print(f"找不到 src.get_bread_roi，將無法顯示綠框: {text}")

# MODEL_FILENAME = "DenseNet121_20260116_205627.pth"
# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "DenseNet", MODEL_FILENAME)

MODEL_PATH = "D:\Project\ML\Bread_classifier\models\ResNet\Bread_20\ResNet18_20260125_194506.pth"

# MODEL_ARCH = "densenet121"
MODEL_ARCH = "resnet18"

root = tk.Tk()
root.title('img choose')
root.geometry('200x200')

print("-" * 50)
try:
    predictor = BreadPredictor(model_path=MODEL_PATH, model_arch=MODEL_ARCH)
except Exception as e:
    print(f"初始化失敗: {e}")
    print("按 Enter 鍵結束程式...")
    input()
    sys.exit(1)
print("-" * 50)

def main():    
    print("\n請選擇一張麵包圖片...")
    img_path = filedialog.askopenfilename(
        title="請選擇圖片",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    if not img_path:
        print("已取消選擇。")
        return

    print(f"讀取圖片: {img_path}")
    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("圖片讀取失敗，格式可能不支援。")
        return

    print("正在辨識麵包種類...")
    eng_name, chi_name, score = predictor.predict(img)

    print("="*30)
    print(f"結果: {chi_name}")
    print(f"英文: {eng_name}")
    print(f"信心: {score:.2f}%")
    print("="*30)

    text_to_show = f"{chi_name} ({score:.1f}%)"
    
    get_bread_roi(img_path, text_to_show)

if __name__ == "__main__":
    while True:
        main()
        q = input("\n按 Enter 繼續測試下一張，輸入 'q' 離開: ")
        if q.lower() == 'q':
            break