import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "ResNet18_Epochs20_2.pth")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20