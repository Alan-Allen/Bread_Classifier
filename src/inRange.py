import FreeSimpleGUI as sg
import numpy as np
import cv2
import os

# --- 設定預設圖 (防呆用) ---
# 建立一個全黑圖片，避免一開始沒有圖可以顯示
img = np.zeros((240, 320, 3), np.uint8)
blurred = cv2.GaussianBlur(img, (7, 7), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# --- 介面佈局 ---
sg.theme('DarkGrey5') # 設定一個比較專業的配色
layout = [
    [sg.Text('影像來源', font=('Arial', 12)), 
     sg.Radio('檔案 (File)', "src", default=True, key='Radio_File', enable_events=True), 
     sg.Radio('攝影機 (Camera)', "src", key='Radio_Cam', enable_events=True)],
    
    [sg.Text('檔案路徑:'), sg.Input(key='filename', size=(30,1)), sg.FileBrowse('瀏覽', target='filename'), sg.Button('載入圖片', key='Load')],
    
    # 影像顯示區 (左邊原圖，右邊遮罩)
    [sg.Image(filename='', key='image', background_color='black'), 
     sg.Image(filename='', key='inRanged', background_color='black')],
    
    [sg.Text('HSV 下限 (Lower)', font=('Arial', 10, 'bold'))],
    [sg.Text('H (色相)'), sg.Slider(range=(0, 180), orientation='h', size=(40, 15), key='H1', enable_events=True),
     sg.Text('S (飽和)'), sg.Slider(range=(0, 255), orientation='h', size=(40, 15), key='S1', enable_events=True),
     sg.Text('V (亮度)'), sg.Slider(range=(0, 255), orientation='h', size=(40, 15), key='V1', enable_events=True)],
    
    [sg.Text('_'*80)], # 分隔線
    
    [sg.Text('HSV 上限 (Upper)', font=('Arial', 10, 'bold'))],
    [sg.Text('H (色相)'), sg.Slider(range=(0, 180), default_value=179, orientation='h', size=(40, 15), key='H2', enable_events=True),
     sg.Text('S (飽和)'), sg.Slider(range=(0, 255), default_value=255, orientation='h', size=(40, 15), key='S2', enable_events=True),
     sg.Text('V (亮度)'), sg.Slider(range=(0, 255), default_value=255, orientation='h', size=(40, 15), key='V2', enable_events=True)],
    
    [sg.Text('目前的參數 (複製這行到程式碼):', text_color='yellow')],
    [sg.InputText(key='Output_Code', size=(80,1), readonly=True)]
]

window = sg.Window('HSV 麵包輪廓調校器', layout, location=(50, 50))
cap = cv2.VideoCapture(0)

# 控制是否使用攝影機
use_camera = False

while True:
    event, values = window.read(timeout=30) # 30ms 更新一次
    
    if event == sg.WIN_CLOSED:
        break

    # 1. 處理輸入源切換
    if event == 'Radio_Cam':
        use_camera = True
    elif event == 'Radio_File':
        use_camera = False

    # 2. 載入圖片 (處理中文路徑)
    if event == 'Load' and not use_camera:
        file_path = values['filename']
        if os.path.exists(file_path):
            try:
                # 使用 imdecode 解決 Windows 中文路徑讀取失敗的問題
                img_raw = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img_raw, (320, 240)) # 統一縮放方便顯示
            except Exception as e:
                sg.popup_error(f"讀取圖片失敗: {e}")
        else:
            sg.popup('找不到檔案，請確認路徑！')

    # 3. 讀取攝影機
    if use_camera:
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (320, 240))

    # --- 影像處理核心 (放在最外面，確保滑桿一拉就變) ---
    # 模糊處理 (去除雜訊，對麵包很重要)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 取得滑桿數值
    h1, s1, v1 = int(values['H1']), int(values['S1']), int(values['V1'])
    h2, s2, v2 = int(values['H2']), int(values['S2']), int(values['V2'])

    # 產生遮罩
    lower = np.array([h1, s1, v1])
    upper = np.array([h2, s2, v2])
    mask = cv2.inRange(hsv_img, lower, upper)

    # 更新圖片到介面
    window['image'].update(data=cv2.imencode('.png', img)[1].tobytes())
    window['inRanged'].update(data=cv2.imencode('.png', mask)[1].tobytes())

    # 自動生成 Python 程式碼字串，方便你複製
    code_text = f"lower = ({h1}, {s1}, {v1}), upper = ({h2}, {s2}, {v2})"
    window['Output_Code'].update(code_text)

cap.release()
window.close()