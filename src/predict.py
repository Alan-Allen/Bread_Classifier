import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import os

from src.config import CLASS_NAMES, CHINESE_MAP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_bread_model.pth")
MODEL_PATH = os.path.join(PROJECT_ROOT, "test", "models", "ResNet18_Epochs20_2.pth")

class BreadPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"載入 AI 模型中... (使用設備: {self.device})")
        
        # 1. 建立模型架構 (ResNet)
        self.model = models.resnet18(pretrained=False) # 預測時不用下載 ImageNet 權重，因為我們要載入自己的
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        # 2. 載入訓練好的權重
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval() # 設定為評估模式 (重要！)
            print("模型載入成功！")
        else:
            print(f"錯誤: 找不到模型檔案 {MODEL_PATH}")
            print("請先執行 train.py 進行訓練！")

        # 3. 定義預處理 (跟訓練時的 val 轉換一樣)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, cv_image):
        """
        輸入: OpenCV 格式的圖片 (BGR)
        輸出: (預測類別名稱, 中文名稱, 信心指數)
        """
        if cv_image is None: return None, None, 0.0

        # 1. OpenCV (BGR) -> PIL (RGB)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # 2. 預處理並增加 Batch 維度
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # 3. 推論
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1) # 轉成百分比
            confidence, predicted_idx = torch.max(probs, 1)

        # 4. 解析結果
        eng_name = CLASS_NAMES[predicted_idx.item()]
        chi_name = CHINESE_MAP.get(eng_name, eng_name)
        score = confidence.item() * 100

        return eng_name, chi_name, score