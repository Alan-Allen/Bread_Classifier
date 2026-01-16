import cv2
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# --- 參數設定 ---
MIN_BREAD_AREA = 2000    # 麵包最小面積
MIN_PLATE_AREA = 10000   # 盤子最小面積
MAX_AREA_RATIO = 0.5
MARGIN_RATIO = 0.05
ASPECT_RATIO_RANGE = (0.2, 5.0) # 寬高比 (支援長條麵包)

def get_bread_roi(img_path, text):
    # 1. 讀取圖片 (支援中文路徑)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: 
        print(f"無法讀取圖片: {img_path}")
        return

    h_img, w_img, _ = img.shape
    img_center_x = w_img // 2
    img_center_y = h_img // 2
    
    # 複製一份用來畫圖
    debug_img = img.copy()

    # ==========================================
    # 1. 第一階段：尋找盤子 (Plate Hunter)
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_plate = cv2.GaussianBlur(gray, (25, 25), 0)
    # 正向二值化 (找白盤子)
    _, binary_plate = cv2.threshold(blur_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_plate, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_bbox = None
    if len(contours_plate) > 0:
        sorted_plates = sorted(contours_plate, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(sorted_plates[0]) > MIN_PLATE_AREA:
            plate_bbox = cv2.boundingRect(sorted_plates[0])

    # 設定搜尋範圍
    if plate_bbox:
        px, py, pw, ph = plate_bbox
        # 畫出盤子框 (藍色)
        cv2.rectangle(debug_img, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
        cv2.putText(debug_img, "Plate", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        search_img = img[py:py+ph, px:px+pw] 
        offset_x, offset_y = px, py        
    else:
        print("未取得盤子位置，切換為全圖搜尋")
        search_img = img
        offset_x, offset_y = 0, 0

    # ==========================================
    # 2. 第二階段：尋找麵包 (Bread Hunter)
    # ==========================================
    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    search_blur = cv2.GaussianBlur(search_gray, (15, 15), 0)
    # 反向二值化 (找深色麵包)
    _, binary_bread = cv2.threshold(search_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    binary_bread = cv2.morphologyEx(binary_bread, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary_bread, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    sh, sw = search_img.shape[:2]
    search_center_x, search_center_y = sw // 2, sh // 2

    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c) 
        
        if area < MIN_BREAD_AREA: continue
        
        aspect_ratio = float(bw) / bh
        if aspect_ratio < ASPECT_RATIO_RANGE[0] or aspect_ratio > ASPECT_RATIO_RANGE[1]:
            continue

        # 中央優先 (相對於搜尋範圍中心)
        obj_center_x = bx + bw / 2
        obj_center_y = by + bh / 2
        dist = np.sqrt((obj_center_x - search_center_x)**2 + (obj_center_y - search_center_y)**2)
        
        candidates.append({'dist': dist, 'bbox': (bx, by, bw, bh)})

    # ==========================================
    # 3. 選出主角並顯示
    # ==========================================
    if len(candidates) > 0:
        candidates.sort(key=lambda k: k['dist'])
        
        winner = candidates[0]
        bx, by, bw, bh = winner['bbox']
        
        # 還原座標
        x = bx + offset_x
        y = by + offset_y
        w = bw
        h = bh
        
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(debug_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print(f"目標鎖定！距離中心: {winner['dist']:.1f} 像素")
    else:
        print("無目標物體被偵測到。")

    # 畫中心點
    cv2.drawMarker(debug_img, (img_center_x, img_center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    # ==========================================
    # 4. 視窗顯示設定 (固定大小)
    # ==========================================
    window_name = 'Detection Result'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 800, 600) # 固定視窗大小為 800x600
    
    cv2.imshow(window_name, debug_img)
    print("按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
