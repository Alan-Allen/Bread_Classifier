import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.config import CLASS_NAMES, CHINESE_MAP
except ImportError:
    print("錯誤: 無法匯入 src.config。")
    sys.exit(1)

class BreadPredictor:
    def __init__(self, model_path, model_arch="densenet121", threshold=70.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"啟動預測引擎... (架構: {model_arch}, 設備: {self.device})")
        
        self.model = self._create_model(model_arch)
        
        self.threshold = threshold
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"成功載入權重: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"權重載入失敗: {e}")
                print("請確認 model_arch 參數是否設定正確？(ResNet 不能裝 DenseNet 的權重)")
        else:
            print(f"找不到檔案: {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _create_model(self, arch):
        num_classes = len(CLASS_NAMES)
        
        if arch == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            
        elif arch == "densenet121":
            model = models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            
        elif arch == "mobilenet_v3":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
            
        else:
            raise ValueError(f"尚未支援的模型架構: {arch}")
            
        return model

    def predict(self, cv_image):
        if cv_image is None: return None, None, 0.0
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        eng_name = CLASS_NAMES[predicted_idx.item()]
        chi_name = CHINESE_MAP.get(eng_name, eng_name)
        score = confidence.item() * 100
        
        if score < self.threshold:
            eng_name = "Unknown"
            chi_name = "未知麵包"
        else:
            eng_name = CLASS_NAMES[predicted_idx.item()]
            chi_name = CHINESE_MAP.get(eng_name, eng_name)

        return eng_name, chi_name, score

if __name__ == "__main__":    
    pth_path = os.path.join(PROJECT_ROOT, "models", "DenseNet", "DenseNet121_20260116_205627.pth")
    predictor = BreadPredictor(model_path=pth_path, model_arch="densenet121")
    
    # pth_path = os.path.join(PROJECT_ROOT, "models", "ResNet18_old.pth")
    # predictor = BreadPredictor(model_path=pth_path, model_arch="resnet18")