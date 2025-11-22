import torch

print("="*70)
print("GPU/CUDA Detection Test")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("\n❌ CUDA is not available!")
    print("\nPossible reasons:")
    print("1. PyTorch was installed without CUDA support (CPU-only version)")
    print("2. CUDA drivers are not installed")
    print("3. GPU is not CUDA-compatible")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("https://pytorch.org/get-started/locally/")


