"""
GPU Compatibility Checker for PyTorch
Helps diagnose CUDA kernel compatibility issues.
"""
import torch
import sys

def check_gpu_compatibility():
    """Check if GPU is compatible with current PyTorch installation."""
    print("=" * 70)
    print("GPU Compatibility Checker")
    print("=" * 70)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"\n✅ CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA is not available. PyTorch was likely installed without CUDA support.")
        print("   Install CUDA-enabled PyTorch from: https://pytorch.org/get-started/locally/")
        return False
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    print(f"✅ GPU Count: {device_count}")
    
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        gpu_name = torch.cuda.get_device_name(i)
        print(f"Name: {gpu_name}")
        
        # Get compute capability
        capability = torch.cuda.get_device_capability(i)
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA Version: {cuda_version}")
        
        # Test basic operations
        print("\nTesting GPU operations...")
        try:
            # Test 1: Simple tensor creation
            test_tensor = torch.zeros(10, device=f'cuda:{i}')
            print("  ✅ Tensor creation: OK")
            
            # Test 2: Simple arithmetic
            result = test_tensor + 1
            print("  ✅ Arithmetic operations: OK")
            
            # Test 3: Matrix multiplication
            a = torch.randn(100, 100, device=f'cuda:{i}')
            b = torch.randn(100, 100, device=f'cuda:{i}')
            c = torch.matmul(a, b)
            print("  ✅ Matrix multiplication: OK")
            
            # Test 4: Memory operations
            del test_tensor, result, a, b, c
            torch.cuda.empty_cache()
            print("  ✅ Memory operations: OK")
            
            print(f"\n✅ GPU {i} is COMPATIBLE with current PyTorch installation!")
            
        except torch.cuda.CudaError as e:
            error_msg = str(e)
            print(f"\n❌ GPU {i} Compatibility Test FAILED!")
            print(f"   Error: {error_msg}")
            
            if "no kernel image" in error_msg.lower():
                print("\n   🔍 DIAGNOSIS: CUDA Kernel Compatibility Issue")
                print("   This means PyTorch was compiled for a different GPU architecture.")
                print("\n   💡 SOLUTIONS:")
                print("   1. Check your GPU's compute capability:")
                print("      Visit: https://developer.nvidia.com/cuda-gpus")
                print("   2. Reinstall PyTorch compatible with your GPU:")
                print("      Visit: https://pytorch.org/get-started/locally/")
                print("      Select the correct CUDA version and compute capability")
                print("   3. Use CPU instead (slower but guaranteed to work):")
                print("      Set device='cpu' in your code")
                print("   4. Try a different PyTorch version")
                return False
            else:
                print(f"\n   Unknown CUDA error. Please check your CUDA installation.")
                return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
    
    print("\n" + "=" * 70)
    print("✅ All GPU compatibility checks passed!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    try:
        success = check_gpu_compatibility()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error during GPU check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

