import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# --- 參數設定 ---
MIN_BREAD_AREA = 2000    # 麵包最小面積
MIN_PLATE_AREA = 10000   # 盤子最小面積
ASPECT_RATIO_RANGE = (0.2, 5.0) # 寬高比

FONT_PATH = "C:/Windows/Fonts/msjh.ttc" 

def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=30):
    if isinstance(img, np.ndarray):
        # OpenCV (BGR) -> PIL (RGB)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, text_size)
    except IOError:
        print(f"警告: 找不到字型 {FONT_PATH}，將使用預設字型 (中文可能無法顯示)")
        font = ImageFont.load_default()

    draw.text((left+2, top+2), text, fill=(0, 0, 0), font=font) 
    draw.text((left, top), text, fill=text_color, font=font)   
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def get_bread_roi(img_path, text):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: 
        print(f"無法讀取圖片: {img_path}")
        return

    h_img, w_img, _ = img.shape
    img_center_x = w_img // 2
    img_center_y = h_img // 2
    
    debug_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_plate = cv2.GaussianBlur(gray, (25, 25), 0)
    
    _, binary_plate = cv2.threshold(blur_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours_plate, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_bbox = None
    if len(contours_plate) > 0:
        sorted_plates = sorted(contours_plate, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(sorted_plates[0]) > MIN_PLATE_AREA:
            plate_bbox = cv2.boundingRect(sorted_plates[0])

    if plate_bbox:
        px, py, pw, ph = plate_bbox
        cv2.rectangle(debug_img, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
        cv2.putText(debug_img, "Plate", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        search_img = img[py:py+ph, px:px+pw] 
        offset_x, offset_y = px, py        
    else:
        print("未取得盤子位置，切換為全圖搜尋")
        search_img = img
        offset_x, offset_y = 0, 0

    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    search_blur = cv2.GaussianBlur(search_gray, (15, 15), 0)
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

        obj_center_x = bx + bw / 2
        obj_center_y = by + bh / 2
        dist = np.sqrt((obj_center_x - search_center_x)**2 + (obj_center_y - search_center_y)**2)
        
        candidates.append({'dist': dist, 'bbox': (bx, by, bw, bh)})

    if len(candidates) > 0:
        candidates.sort(key=lambda k: k['dist'])
        winner = candidates[0]
        bx, by, bw, bh = winner['bbox']
        
        x = bx + offset_x
        y = by + offset_y
        w = bw
        h = bh
        
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        text_y = max(0, y - 40) 
        debug_img = cv2_img_add_text(debug_img, text, x, text_y, (0, 255, 0), 30)
        
        print(f"ROI 鎖定成功！")
    else:
        print("未偵測到明顯物體，顯示原始圖片。")
        debug_img = cv2_img_add_text(debug_img, text, 10, 10, (0, 255, 0), 30)

    cv2.drawMarker(debug_img, (img_center_x, img_center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    window_name = 'Detection Result'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 800, 600)
    
    cv2.imshow(window_name, debug_img)
    print("請按任意鍵關閉圖片視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()