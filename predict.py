import torch
from PIL import Image
import cv2
import os
import sys

from src import Config, create_model, get_transforms

class BreadPredictor:
    def __init__(self, model_path, model_arch=Config.MODEL_NAME, threshold=70.0):
        self.device = Config.device
        print(f"啟動預測引擎... (架構: {model_arch}, 設備: {self.device})")
        
        self.model = create_model(model_arch, num_classes=len(Config.CLASS_NAMES), pretrained=False)
        
        self.threshold = threshold
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True) 
                print(f"成功載入權重: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"權重載入失敗: {e}")
                print("請確認 model_arch 參數是否與權重檔匹配？(例如用 ResNet 代碼去載 DenseNet 權重)")
        else:
            print(f"找不到權重檔案: {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(is_training=False)

    def predict(self, cv_image):
        if cv_image is None: return "Unknown", "讀取失敗", 0.0
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        score = confidence.item() * 100
        idx = predicted_idx.item()
        
        if score < self.threshold:
            print(f"信心 < {self.threshold}")
            eng_name = "Unknown"
            chi_name = "未知麵包"
        else:
            eng_name = Config.CLASS_NAMES[idx]
            chi_name = getattr(Config, 'CHINESE_MAP', {}).get(eng_name, eng_name)
            
        return eng_name, chi_name, score

if __name__ == "__main__":    
    #pth_path = os.path.join(Config.MODEL_SAVE_DIR, "DenseNet121_20260126_182735.pth")
    pth_path = r"D:\\Project\\ML\\Bread_classifier\\models\\resnet\\Bread_20\\ResNet18_20260125_194506.pth"
    
    if os.path.exists(pth_path):
        predictor = BreadPredictor(model_path=pth_path, model_arch=Config.MODEL_NAME)
        print("初始化成功，等待影像輸入...")
    else:
        print(f"測試檔案不存在: {pth_path}")