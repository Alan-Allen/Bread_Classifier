import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    print("正在計算混淆矩陣，請稍候...")
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
    plt.ylabel('True Label (真實類別)')
    plt.xlabel('Predicted Label (預測類別)')
    plt.title('Confusion Matrix')
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
    print(f"總樣本數 (Total Samples): {len(all_labels)}")
    print(f"Top-1 Acc               : {top1_acc*100:.2f}%")
    print(f"Top-3 Acc               : {topk_acc*100:.2f}%")
    print(f"Macro Precision         : {precision:.4f}")
    print(f"Macro Recall            : {recall:.4f}")
    print(f"Macro F1 Score          : {f1:.4f}")
    print(f"--------------------------------------------------")

    print(f"\n[各類別詳細報告 Per-Class Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("="*50 + "\n")
    
    
def save_experiment_results(save_dir, config, model_name, pretrained, train_time_str, dataset_sizes, best_metrics, full_report_dict):
    report_dir = os.path.join(save_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_data = {
        "Date": timestamp,
        "Model_Arch": model_name,
        "Pretrained": str(pretrained),
        "Epochs": getattr(config, 'EPOCHS', 'N/A'),
        "Batch_Size": getattr(config, 'BATCH_SIZE', 'N/A'),
        "Learning_Rate": getattr(config, 'LEARNING_RATE', 'N/A'),
        "Num_Classes": len(getattr(config, 'CLASS_NAMES', [])),
        "Train_Size": dataset_sizes.get('train', 0),
        "Val_Size": dataset_sizes.get('val', 0),
        "Best_Val_Acc": f"{best_metrics.get('best_acc', 0):.4f}",
        "Top1_Acc": f"{best_metrics.get('top1', 0):.4f}",
        "Top3_Acc": f"{best_metrics.get('top3', 0):.4f}",
        "Macro_F1": f"{best_metrics.get('macro_f1', 0):.4f}",
        "Min_Val_Loss": f"{best_metrics.get('min_loss', 0):.4f}",
        "Train_Time": train_time_str,
        "Device": str(getattr(config, 'DEVICE', 'cpu'))
    }

    csv_path = os.path.join(save_dir, f"training_summary_{file_safe_time}.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)
        
    print(f"實驗摘要已寫入: {csv_path}")

    txt_filename = f"Report_{model_name}_{file_safe_time}.txt"
    txt_path = os.path.join(report_dir, txt_filename)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Experiment Report: {model_name} ===\n")
        f.write(f"Date: {timestamp}\n")
        f.write("-" * 50 + "\n")
        
        for k, v in summary_data.items():
            f.write(f"{k:<15}: {v}\n")
            
        f.write("-" * 50 + "\n")
        f.write("=== Per-Class Performance (各類別詳細數據) ===\n\n")
        
        df = pd.DataFrame(full_report_dict).transpose()
        if 'support' in df.columns:
            df['support'] = df['support'].astype(int)
        
        f.write(df.to_string())
        f.write("\n\n" + "="*50)

    print(f"詳細報告已儲存: {txt_path}")