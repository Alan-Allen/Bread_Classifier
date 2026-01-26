import os
import torch
import matplotlib.pyplot as plt

class Config:
    MODEL_NAME = "resnet18"
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', MODEL_NAME)
    
    NUM_CLASSES = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用運算設備: {device}")
    
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    RAW_BREAD_MAP = {
        "丹麥咖啡奶酥波羅": "danish_coffee_polo",
        "丹麥奶酥波蘿": "danish_butter_polo",
        "丹麥波蘿": "danish_polo",
        "核桃布里奧斯": "walnut_brioche",
        "白麵包-巧克力": "white_bread_choc",
        "白麵包-草莓": "white_bread_straw",
        "肉鬆沙拉": "rousong_salad",
        "花生麵包": "peanut_bread",
        "起酥肉鬆": "puff_rousong",
        "香蒜巴西里": "garlic_parsley",
        "活力三明治": "energy_sandwich",
        "香草杯子": "vanilla_cupcake",
        "特濃奶霜蛋糕捲-巧克力": "choc_cream_roll",
        "脆皮奶油": "crispy_butter_bun",
        "脆皮麻花": "crispy_twist",
        "脆脆烤年糕": "crispy_rice_cake",
        "草莓夾心": "strawberry_filling",
        "草莓雪紡蛋糕": "strawberry_chiffon",
        "起司辣蛋糕": "spicy_cheese_cake",
        "起司燻雞可頌": "smoked_chicken_croissant",
    }
    
    CLASS_NAMES = sorted(list(RAW_BREAD_MAP.values()))
    CHINESE_MAP = {v: k for k, v in RAW_BREAD_MAP.items()}