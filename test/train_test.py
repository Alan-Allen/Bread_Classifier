import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_filename = f"{timestamp}.pth"

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "test", "models", model_filename)


def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n正在繪製混淆矩陣圖表...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真實類別 (True Label)', fontsize=14)
    plt.xlabel('預測類別 (Predicted Label)', fontsize=14)
    plt.title('混淆矩陣 (Confusion Matrix)', fontsize=16)
    plt.show()

def evaluate_detailed_metrics(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n" + "="*50)
    print("正在進行深度評估 (計算 Top-K, F1, Recall...)...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = torch.cat(all_probs)
    all_labels_tensor = torch.tensor(all_labels).to(device)

    top1_correct = (all_probs.topk(1, dim=1)[1] == all_labels_tensor.view(-1, 1)).sum().item()
    top1_acc = top1_correct / len(all_labels)
    
    k = 3
    if len(class_names) >= k:
        topk_correct = (all_probs.topk(k, dim=1)[1] == all_labels_tensor.view(-1, 1)).sum().item()
        topk_acc = topk_correct / len(all_labels)
    else:
        topk_acc = 1.0

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n[整體效能指標 Summary]")
    print(f"--------------------------------------------------")
    print(f"準確率 (Accuracy)      : {top1_acc*100:.2f}%")
    print(f"Top-3 準確率           : {topk_acc*100:.2f}%")
    print(f"宏觀平均 F1 (Macro F1) : {f1:.4f}")
    print(f"--------------------------------------------------")

    print(f"\n[各類別詳細報告 Per-Class Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("="*50 + "\n")
    
    return top1_acc, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用運算設備: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("正在讀取資料...")
    
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"錯誤: 找不到訓練資料夾，請確認 {os.path.join(DATA_DIR, 'train')} 是否存在且有圖片。")
        return

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"資料載入成功！")
    print(f"類別數量: {len(class_names)} 種")
    print(f"類別清單: {class_names}")
    print(f"訓練集: {dataset_sizes['train']} 張 | 驗證集: {dataset_sizes['val']} 張")

    print("\n正在下載並建立 ResNet18 模型...")
    model = models.resnet18(pretrained=True)
    
    num_ftrs = model.fc.in_features 
    model.fc = nn.Linear(num_ftrs, len(class_names)) 

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"\n開始訓練: 共 {EPOCHS} 輪")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n訓練完成！總耗時: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已儲存至: {MODEL_SAVE_PATH}")
    
    evaluate_detailed_metrics(model, dataloaders['val'], class_names, device)

    plot_confusion_matrix(model, dataloaders['val'], class_names, device)

if __name__ == '__main__':
    main()