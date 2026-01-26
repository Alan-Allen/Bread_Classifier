import os
import torch
from torchvision import datasets
from .config import Config
from .transforms import get_transforms

def create_dataloaders():
    train_dir = os.path.join(Config.DATA_DIR, 'train')
    val_dir = os.path.join(Config.DATA_DIR, 'val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"找不到訓練資料夾: {train_dir}")

    #Dataset
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, get_transforms(is_training=True)),
        'val': datasets.ImageFolder(val_dir, get_transforms(is_training=False))
    }

    #DataLoader
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    }

    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes, class_names