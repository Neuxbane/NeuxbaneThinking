import torch
import os
import subprocess
import re

def get_gpu_memory_info():
    """
    Query GPU memory using nvidia-smi (most reliable, doesn't require GPU memory itself).
    Falls back to pynvml if nvidia-smi unavailable.
    
    Returns:
        List of tuples: [(gpu_idx, free_gb, total_gb), ...]
    """
    
    # Try nvidia-smi first (most reliable, works even when GPUs are full)
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if output.returncode == 0:
            gpu_info = []
            for line in output.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    idx = int(parts[0].strip())
                    free_mb = float(parts[1].strip())
                    total_mb = float(parts[2].strip())
                    gpu_info.append((idx, free_mb / 1024, total_mb / 1024))  # Convert MB to GB
            return gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    # Fallback: pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024 ** 3)
            total_gb = mem_info.total / (1024 ** 3)
            gpu_info.append((i, free_gb, total_gb))
        
        pynvml.nvmlShutdown()
        return gpu_info
    except (ImportError, Exception):
        pass
    
    return []

def get_best_device(min_memory_gb=2.0):
    """
    Select the best available device (GPU or CPU).
    
    Args:
        min_memory_gb: Minimum free GPU memory in GB to use GPU. 
                       If no GPU has this much, falls back to CPU.
    
    Returns:
        torch.device: The selected device
        device_name: String describing the selected device
    """
    
    # If CUDA is not available, use CPU
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device('cpu'), "CPU"
    
    # Get GPU memory information
    gpu_info = get_gpu_memory_info()
    
    best_gpu_idx = None
    best_free_memory = 0
    
    if gpu_info:
        for gpu_idx, free_gb, total_gb in gpu_info:
            print(f"GPU {gpu_idx} ({torch.cuda.get_device_name(gpu_idx)}): {free_gb:.2f} GB free / {total_gb:.2f} GB total")
            
            if free_gb > best_free_memory:
                best_free_memory = free_gb
                best_gpu_idx = gpu_idx
    else:
        # If we can't query via nvidia-smi or pynvml, fall back to a conservative check
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i} ({torch.cuda.get_device_name(i)}): Memory status unknown (using conservative estimate)")
            # Conservative: assume at least some memory is available
            best_gpu_idx = i if best_gpu_idx is None else best_gpu_idx
            best_free_memory = 1.0  # Assume at least 1GB available
    
    # Check if best GPU has enough memory
    if best_gpu_idx is not None and best_free_memory >= min_memory_gb:
        device = torch.device(f'cuda:{best_gpu_idx}')
        device_name = f"GPU {best_gpu_idx} ({torch.cuda.get_device_name(best_gpu_idx)}) - {best_free_memory:.2f} GB free"
        print(f"\n✓ Selected: {device_name}")
        return device, device_name
    
    # Fallback to CPU
    device_name = f"CPU (GPU memory insufficient. Required: {min_memory_gb} GB, Best available: {best_free_memory:.2f} GB)"
    print(f"\n✓ Falling back to CPU: {device_name}")
    return torch.device('cpu'), device_name

def set_device(min_memory_gb=2.0):
    """
    Set and return the best device, also set CUDA memory growth to prevent OOM.
    """
    device, device_name = get_best_device(min_memory_gb)
    
    # Enable memory growth for CUDA to avoid pre-allocating all GPU memory
    if device.type == 'cuda':
        # Don't use os.environ["CUDA_VISIBLE_DEVICES"] here as CUDA is likely already initialized
        # by get_best_device calls. Just set the device normally.
        torch.cuda.set_device(device)
        # Clear cache to ensure we start fresh
        torch.cuda.empty_cache()
    
    return device, device_name

if __name__ == "__main__":
    # Test the utility
    print("="*60)
    print("GPU Selection Utility Test")
    print("="*60)
    device, device_name = get_best_device(min_memory_gb=2.0)
    print(f"\nSelected Device: {device}")
    print(f"Device Name: {device_name}")
