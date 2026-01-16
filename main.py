import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

from get_bread_roi import get_bread_roi
from src.predict import BreadPredictor
    
root = tk.Tk()
root.title('img choose')
root.geometry('200x200')

predictor = BreadPredictor()

def main():
    #img_path = r'D:\Project\ML\Bread_classifier\data\test\danish_butter_polo\opencv_20250801_184847_684930.png'
    
    img_path = filedialog.askopenfilename(
        title="請選擇圖片",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    if not img_path:
        return

    print(f"\n讀取圖片: {img_path}")
    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("圖片讀取失敗")
        return

    print("正在辨識麵包種類...")
    eng_name, chi_name, score = predictor.predict(img)

    print("="*30)
    print(f"結果: {chi_name} ({eng_name})")
    print(f"信心: {score:.2f}%")
    print("="*30)

    text = f"{eng_name} ({score:.1f}%)"
    
    get_bread_roi(img_path, text)

    print("按任意鍵離開...")

if __name__ == "__main__":
    main()