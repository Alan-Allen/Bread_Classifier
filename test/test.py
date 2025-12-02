import torch
import time

def check_pytorch_environment():
    print("=" * 30)
    print("PyTorch 環境檢測開始")
    print("=" * 30)

    # 1. 檢查版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 2. 檢查 CUDA (GPU) 是否可用
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"CUDA (GPU) 狀態: 可用 (太好了！訓練會很快)")
        device_count = torch.cuda.device_count()
        print(f"   偵測到 {device_count} 張顯示卡:")
        for i in range(device_count):
            print(f"   [{i}] {torch.cuda.get_device_name(i)}")
        
        # 設定使用 GPU
        device = torch.device("cuda")
    else:
        print(f"CUDA (GPU) 狀態: 未偵測到 (將使用 CPU 訓練，速度會較慢)")
        print("(如果你有 NVIDIA 顯卡，可能是驅動程式或 PyTorch 版本沒裝對)")
        device = torch.device("cpu")

    print("-" * 30)
    print(f"正在測試運算效能 (使用設備: {device})...")

    # 3. 實際運算測試 (矩陣乘法)
    try:
        # 建立兩個隨機的大矩陣 (3000 x 3000)
        x = torch.rand(3000, 3000).to(device)
        y = torch.rand(3000, 3000).to(device)

        start_time = time.time()
        # 執行矩陣乘法
        result = torch.matmul(x, y)
        end_time = time.time()

        print(f"運算測試成功！")
        print(f"耗時: {end_time - start_time:.4f} 秒")
        
        if cuda_available:
             print("\n恭喜！你的環境已經準備好可以用 GPU 訓練麵包辨識模型了！")
        else:
             print("\n環境運作正常 (CPU 模式)。可以開始寫程式了！")

    except Exception as e:
        print(f"\n運算測試失敗: {e}")

if __name__ == "__main__":
    check_pytorch_environment()