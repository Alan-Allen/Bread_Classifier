from torchvision import transforms

def get_transforms(is_training=False):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)), #強制縮放
            transforms.RandomHorizontalFlip(), # 左右翻轉
            transforms.RandomRotation(10),     # 稍微旋轉 +/- 10度
            # transforms.ColorJitter(brightness=0.1), # 改變亮度
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])