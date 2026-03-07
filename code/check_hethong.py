import torch
import torchvision
import sys

def main():
    print("===== System Info =====")
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} name:", torch.cuda.get_device_name(i))
            print(f"  Capability:", torch.cuda.get_device_capability(i))
            print(f"  Memory Allocated:", torch.cuda.memory_allocated(i))
            print(f"  Memory Cached:", torch.cuda.memory_reserved(i))

    print("===== AMP check =====")
    try:
        x = torch.randn(3, 3, device="cuda", dtype=torch.float16)
        y = torch.randn(3, 3, device="cuda", dtype=torch.float16)
        z = x @ y
        print("AMP (float16 matmul) works")
    except Exception as e:
        print("AMP test failed:", e)

if __name__ == "__main__":
    main()