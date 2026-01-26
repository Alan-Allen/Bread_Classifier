import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    return top1_acc, f1