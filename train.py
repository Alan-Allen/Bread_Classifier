import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
from datetime import datetime
from sklearn.metrics import classification_report

from src import Config
from src.dataset import create_dataloaders
from src.model_factory import create_model
from src.train_engine import train_one_epoch, validate_one_epoch
from src.utils import plot_confusion_matrix, evaluate_detailed_metrics, save_experiment_results

def main():
    print(f"使用運算設備: {Config.device}")
    
    print("正在讀取資料...")
    dataloaders, dataset_sizes, class_names = create_dataloaders()
    print(f"資料載入成功！類別數量: {len(class_names)} 種")
    print(f"訓練集: {dataset_sizes['train']} 張 | 驗證集: {dataset_sizes['val']} 張")

    print(f"正在建立 {Config.MODEL_NAME} 模型...")
    model = create_model(Config.MODEL_NAME, num_classes=len(class_names)).to(Config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"\n開始訓練: 共 {Config.EPOCHS} 輪")
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_val_loss = 1.0

    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, Config.device
        )
        exp_lr_scheduler.step()
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        val_loss, val_acc = validate_one_epoch(
            model, dataloaders['val'], criterion, Config.device
        )
        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("發現更準的模型！暫存中...")
        
        if val_loss < min_val_loss:
            best_acc = val_loss
            

    time_elapsed = time.time() - since
    train_time_str = f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    print(f'\n訓練完成！總耗時: {train_time_str}')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{Config.MODEL_NAME}_{timestamp}.pth"
    save_path = os.path.join(Config.MODEL_SAVE_DIR, save_filename)
    
    torch.save(model.state_dict(), save_path)
    print(f"模型已儲存至: {save_path}")

    evaluate_detailed_metrics(model, dataloaders['val'], class_names, Config.device)
    plot_confusion_matrix(model, dataloaders['val'], class_names, Config.device)
    
    print("\n" + "="*50)
    print("\n正在進行最終數據結算...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.append(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_probs = torch.cat(all_probs)
    all_labels_tensor = torch.tensor(all_labels).to(Config.device)
    
    top1 = (all_probs.topk(1, dim=1)[1] == all_labels_tensor.view(-1, 1)).sum().item() / len(all_labels)
    top3 = (all_probs.topk(3, dim=1)[1] == all_labels_tensor.view(-1, 1)).sum().item() / len(all_labels)
    
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    best_metrics = {
        'best_acc': best_acc,
        'min_loss': min_val_loss,
        'top1': top1,
        'top3': top3,
        'macro_f1': report_dict['macro avg']['f1-score']
    }
    
    save_experiment_results(
        save_dir=os.path.join(Config.PROJECT_ROOT, "Log"),
        config=Config,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        train_time_str=train_time_str,
        dataset_sizes=dataset_sizes,
        best_metrics=best_metrics,
        full_report_dict=report_dict
    )

if __name__ == '__main__':
    main()