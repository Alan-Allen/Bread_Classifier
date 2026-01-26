import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
from datetime import datetime

from src.config import Config
from src.dataset import create_dataloaders
from src.model_factory import create_model
from src.train_engine import train_one_epoch, validate_one_epoch
from src.utils import plot_confusion_matrix, evaluate_detailed_metrics

def main():
    print(f"使用運算設備: {Config.DEVICE}")
    
    print("正在讀取資料...")
    dataloaders, dataset_sizes, class_names = create_dataloaders()
    print(f"資料載入成功！類別數量: {len(class_names)} 種")
    print(f"訓練集: {dataset_sizes['train']} 張 | 驗證集: {dataset_sizes['val']} 張")

    print(f"正在建立 {Config.MODEL_NAME} 模型...")
    model = create_model(Config.Model_name, num_classes=len(class_names)).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"\n開始訓練: 共 {Config.EPOCHS} 輪")
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, Config.DEVICE
        )
        exp_lr_scheduler.step() # 更新 Learning Rate
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        val_loss, val_acc = validate_one_epoch(
            model, dataloaders['val'], criterion, Config.DEVICE
        )
        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("發現更準的模型！暫存中...")

    time_elapsed = time.time() - since
    print(f'\n訓練完成！總耗時: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{Config.MODEL_NAME}_{timestamp}.pth"
    save_path = os.path.join(Config.MODEL_SAVE_DIR, save_filename)
    
    torch.save(model.state_dict(), save_path)
    print(f"模型已儲存至: {save_path}")

    evaluate_detailed_metrics(model, dataloaders['val'], class_names, Config.DEVICE)
    plot_confusion_matrix(model, dataloaders['val'], class_names, Config.DEVICE)

if __name__ == '__main__':
    main()