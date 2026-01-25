RAW_BREAD_MAP = {
    # --- 原有 10 類 ---
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
    
    # --- 新增 10 類 ---
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

# 英文類別清單
CLASS_NAMES = sorted(list(RAW_BREAD_MAP.values()))

# 反向對照表 (英文 -> 中文)
CHINESE_MAP = {v: k for k, v in RAW_BREAD_MAP.items()}

# CLASS_NAMES = sorted([
#     "danish_coffee_polo",
#     "danish_butter_polo",
#     "danish_polo",
#     "walnut_brioche",
#     "white_bread_choc",
#     "white_bread_straw",
#     "rousong_salad",
#     "peanut_bread",
#     "puff_rousong",
#     "garlic_parsley",
# ])

# CHINESE_MAP = {
#     "danish_coffee_polo": "丹麥咖啡奶酥波羅",
#     "danish_butter_polo": "丹麥奶酥波蘿",
#     "danish_polo": "丹麥波蘿",
#     "walnut_brioche": "核桃布里奧斯",
#     "white_bread_choc": "白麵包-巧克力",
#     "white_bread_straw": "白麵包-草莓",
#     "rousong_salad": "肉鬆沙拉",
#     "peanut_bread": "花生麵包",
#     "puff_rousong": "起酥肉鬆",
#     "garlic_parsley": "香蒜巴西里",
# }