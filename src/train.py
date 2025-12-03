import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "best_bread_model.pth")

BATCH_SIZE = 32  
LEARNING_RATE = 0.001
EPOCHS = 15      

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用運算設備: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # 隨機裁切並縮放 (模擬麵包遠近不同)
            transforms.RandomHorizontalFlip(), # 隨機左右翻轉 (麵包左轉右轉都是麵包)
            transforms.ToTensor(),             # 轉成電腦看得懂的數字矩陣
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 標準化
        ]),
        
        'val': transforms.Compose([
            transforms.Resize(256),            # 驗證時只要縮放
            transforms.CenterCrop(224),        # 取中間
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
    print(f"   類別數量: {len(class_names)} 種")
    print(f"   類別清單: {class_names}")
    print(f"   訓練集: {dataset_sizes['train']} 張 | 驗證集: {dataset_sizes['val']} 張")

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
                print("發現更準的模型！暫存中...")

    time_elapsed = time.time() - since
    print(f'\n訓練完成！總耗時: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已儲存至: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()