import torch
import os
import subprocess
import re
import time
import signal
import sys

# Global state for the Cooperative Handover
HANDOVER_REQUESTED = False
HANDOVER_COMPLETE = False

def handle_handover_request(signum, frame):
    global HANDOVER_REQUESTED
    HANDOVER_REQUESTED = True

def handle_handover_success(signum, frame):
    global HANDOVER_COMPLETE
    HANDOVER_COMPLETE = True

def get_gpu_memory_info():
    """
    Query GPU info using nvidia-smi (most reliable, doesn't require GPU memory itself).
    
    Returns:
        List of tuples: [(gpu_idx, free_gb, total_gb, name, is_errored), ...]
    """
    
    # 1. Detection of 'ERR!' in plain nvidia-smi
    errored_indices = set()
    try:
        smi_out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if smi_out.returncode == 0:
            # Each GPU block is usually surrounded by +---+ separators.
            # We split by these to isolate each GPU's data.
            blocks = re.split(r'\+[-+]+\+', smi_out.stdout)
            for block in blocks:
                # Look for a line starting with '|' and a number (the index)
                # and contains some vendor name like 'NVIDIA' and status like 'On'/'Off'.
                header_match = re.search(r'\|\s+(\d+)\s+[\w\s.-]+\s+(?:On|Off)\s+\|', block)
                if header_match:
                    idx = int(header_match.group(1))
                    # Check if 'ERR!' appears anywhere in this block (Power, Temp, Fan, etc.)
                    if "ERR!" in block:
                        errored_indices.add(idx)
    except:
        pass

    # 2. Main query for memory and name using CSV format
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total,name', '--format=csv,nounits,noheader'],
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
                    name = parts[3].strip()
                    is_err = idx in errored_indices
                    gpu_info.append((idx, free_mb / 1024, total_mb / 1024, name, is_err))  # Convert MB to GB
            return gpu_info
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    return []

def get_lock_file(gpu_idx=None):
    """Get the path to the reservation lock file for the current user and GPU index."""
    user = os.environ.get('USER', 'default_user')
    if gpu_idx is not None:
        return f"/tmp/gpu_reserver_{user}_{gpu_idx}.pid"
    return f"/tmp/gpu_reserver_{user}_*"

def release_any_reservations(target_gpu_idx=None):
    """
    Cooperatively request reservations to release memory (using SIGUSR1).
    Returns list of PIDs that were signaled.
    """
    import glob
    if target_gpu_idx is not None:
        pattern = get_lock_file(target_gpu_idx)
    else:
        pattern = get_lock_file()
        
    lock_files = glob.glob(pattern)
    signaled_pids = []
    
    if not lock_files:
        return []
        
    for lock_file in lock_files:
        try:
            with open(lock_file, 'r') as f:
                data = f.read().strip().split(':')
                if len(data) >= 2:
                    pid = int(data[0])
                    gpu_idx = int(data[1])
                    
                    print(f"Requesting cooperative release from GPU {gpu_idx} (PID {pid})...")
                    try:
                        os.kill(pid, signal.SIGUSR1) # SIGUSR1 = Request Release
                        signaled_pids.append(pid)
                    except ProcessLookupError:
                        pass
            os.remove(lock_file)
        except Exception:
            if os.path.exists(lock_file):
                os.remove(lock_file)
    
    return signaled_pids

def reserve_gpu_memory(target_gb=8.0, gpu_idx=None, wait_interval=5):
    """
    Wait until ANY GPU memory is available, then start grabbing it chunk-by-chunk 
    until target_gb is reached.
    """
    target_desc = f"GPU {gpu_idx}" if gpu_idx is not None else "ANY GPU"
    print(f"Incremental Reserver: Target {target_gb:.2f} GB on {target_desc}")
    
    while True:
        gpu_info = get_gpu_memory_info()
        # candidate is (idx, free, total, name, is_err)
        candidates = [g for g in gpu_info if (gpu_idx is None or g[0] == gpu_idx) and not g[4]]
            
        if not candidates:
            print(f"Error: No suitable GPUs found (or all have ERR!/insufficient memory).")
            return False
            
        # Pick the one with the most free memory to start our aggregration
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_gpu = candidates[0]
        
        # Only start if we can get at least 200MB to avoid useless squatting
        if selected_gpu[1] < 0.2:
            print(f"Waiting for any GPU to have >200MB free... (Best: {selected_gpu[1]:.2f} GB)")
            time.sleep(wait_interval)
            continue

        idx = selected_gpu[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        # After setting CUDA_VISIBLE_DEVICES, the selected GPU is always at index 0
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
        reserved_blocks = []
        current_sum_gb = 0
        
        print(f"Starting aggregation on GPU {idx}...")
        
        try:
            # Write PID, GPU index, and initial amount to lock file
            lock_file = get_lock_file(idx)
            with open(lock_file, 'w') as f:
                f.write(f"{os.getpid()}:{idx}:{current_sum_gb}")

            global HANDOVER_REQUESTED, HANDOVER_COMPLETE
            HANDOVER_REQUESTED = False
            HANDOVER_COMPLETE = False
            
            def terminate_handler(signum, frame):
                print("\nReservation shutdown signal received.")
                if os.path.exists(lock_file): os.remove(lock_file)
                sys.exit(0)
            
            signal.signal(signal.SIGTERM, terminate_handler)
            signal.signal(signal.SIGINT, terminate_handler)
            signal.signal(signal.SIGUSR1, handle_handover_request)
            signal.signal(signal.SIGUSR2, handle_handover_success)

            while True:
                if HANDOVER_REQUESTED:
                    print(f"Handover requested. Releasing {len(reserved_blocks)} blocks...")
                    # Release one by one to let the script grab them
                    while reserved_blocks:
                        reserved_blocks.pop()
                        torch.cuda.empty_cache()
                        time.sleep(0.1) # Small gap for script to catch it
                    
                    print("Waiting for script confirmation (SIGUSR2)...")
                    while not HANDOVER_COMPLETE:
                        time.sleep(1)
                    
                    # Final cleanup just in case
                    time.sleep(30)
                    
                    print("✓ Handover finished. Re-seeking new GPU...")
                    HANDOVER_COMPLETE = False
                    if os.path.exists(lock_file): os.remove(lock_file)
                    break # Back to outer loop search

                # Health Check: If the GPU we are on develops an error, move on
                info = get_gpu_memory_info()
                target_data = next((g for g in info if g[0] == idx), None)
                if not target_data or target_data[4]:
                    print(f"\n[!] GPU {idx} reported an error or is no longer available. Discarding reservation...")
                    if os.path.exists(lock_file): os.remove(lock_file)
                    # Clear blocks to actually free the memory (if possible)
                    reserved_blocks.clear()
                    torch.cuda.empty_cache()
                    break # Back to outer loop to find a healthy GPU

                # Normal Aggregation Logic
                if current_sum_gb < target_gb:
                    # Check how much is currently free
                    try:
                        # Use already fetched info
                        current_free = target_data[1]
                        
                        # Grab whatever is free, leaving a 100MB buffer for system stability
                        to_grab = min(current_free - 0.1, target_gb - current_sum_gb)
                        if to_grab > 0.05: # At least 50MB chunks
                            num_elements = int((to_grab * 0.98 * (1024**3)) / 4)
                            block = torch.zeros((num_elements,), device=device, dtype=torch.float32)
                            reserved_blocks.append(block)
                            current_sum_gb += (to_grab * 0.98)
                            
                            # Update lock file
                            with open(lock_file, 'w') as f:
                                f.write(f"{os.getpid()}:{idx}:{current_sum_gb:.2f}")
                            print(f"  [GPU {idx}] Aggregated block: {to_grab * 0.98:.2f} GB | Total: {current_sum_gb:.2f}/{target_gb:.2f}")
                    except Exception:
                        pass # nvidia-smi spike?

                time.sleep(2)
            
            # If we broke the inner loop because of a handover
            continue

        except torch.cuda.OutOfMemoryError:
            print("Aggregation reached physical limit. Holding current amount.")
            time.sleep(10)
        except Exception as e:
            print(f"Error on GPU {idx}: {e}")
            if os.path.exists(lock_file): os.remove(lock_file)
            print("Seeking another healthy GPU...")
            time.sleep(2)
            continue

def get_best_device(min_memory_gb=2.0, release_reserved=True):
    """
    Select the best available device (GPU or CPU).
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device('cpu'), "CPU"
    
    def get_status():
        gpu_info = get_gpu_memory_info()
        import glob
        pattern = get_lock_file()
        lock_files = glob.glob(pattern)
        user_reservations = {}
        for lock_file in lock_files:
            try:
                with open(lock_file, 'r') as f:
                    data = f.read().strip().split(':')
                    if len(data) >= 2:
                        idx = int(data[1])
                        amount = float(data[2]) if len(data) >= 3 else 0
                        user_reservations[idx] = user_reservations.get(idx, 0) + amount
            except: pass
        return gpu_info, user_reservations

    # 1. Scan current status
    gpu_info, user_reservations = get_status()
    
    best_gpu_idx = None
    best_effective_free = -1
    best_real_free = 0
    best_gpu_name = "Unknown"
    
    if gpu_info:
        print("-" * 30)
        for gpu_idx, free_gb, total_gb, name, is_err in gpu_info:
            reserved_by_me = user_reservations.get(gpu_idx, 0)
            effective_free = free_gb + reserved_by_me
            
            # Print for user visibility
            if is_err:
                status = "ERR!"
            else:
                status = "OK" if effective_free >= min_memory_gb else "LOW VRAM"
            
            res_str = f" [RESERVED: {reserved_by_me:.2f} GB]" if reserved_by_me > 0 else ""
            print(f"GPU {gpu_idx} ({name}): {free_gb:.2f} GB real / {effective_free:.2f} GB effective {res_str} [{status}]")
            
            # Selection criteria: must not be errored
            if not is_err and effective_free > best_effective_free:
                best_effective_free = effective_free
                best_real_free = free_gb
                best_gpu_idx = gpu_idx
                best_gpu_name = name
        print("-" * 30)
    else:
        # Fallback to internal torch check
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            best_gpu_idx = 0
            best_effective_free = 1.0 # placeholder
            best_real_free = 1.0
            best_gpu_name = f"GPU 0"

    # 2. Decision Logic
    if best_gpu_idx is not None:
        # If we have enough EFFECTIVE memory but not enough REAL memory, kill the reservation
        needs_release = release_reserved and (best_real_free < min_memory_gb) and (best_effective_free >= min_memory_gb)
        
        # Or if we strictly don't have enough even with reservation, try clearing all anyway just in case
        force_clean = release_reserved and (best_effective_free < min_memory_gb)
        
        if needs_release or (force_clean and user_reservations):
            print(f"Triggering Handover Request on GPU {best_gpu_idx} to meet {min_memory_gb:.2f} GB requirement...")
            signaled_pids = release_any_reservations(best_gpu_idx if needs_release else None)
            
            # PROTECTIVE AGGREGATION: Grab memory as soon as it releases to prevent others from stealing it
            print(f"Securing {min_memory_gb:.2f} GB by protective aggregation...")
            captured_blocks = []
            captured_gb = 0
            device_securing = torch.device(f'cuda:{best_gpu_idx}')
            
            max_retries = 40 # 20 seconds total wait
            for i in range(max_retries):
                needed_gb = min_memory_gb - (captured_gb * 0.95) # allow for small rounding
                if needed_gb <= 0:
                    break
                    
                # Refresh status
                info_now, _ = get_status()
                target_gpu = next((g for g in info_now if g[0] == best_gpu_idx), None)
                
                if target_gpu:
                    real_free = target_gpu[1]
                    # Attempt to grab a chunk if at least 100MB is free
                    to_grab = min(real_free - 0.05, needed_gb)
                    if to_grab > 0.05:
                        try:
                            num_els = int((to_grab * 0.98 * (1024**3)) / 4)
                            # Secure the memory for THIS process
                            block = torch.zeros((num_els,), device=device_securing, dtype=torch.float32)
                            captured_blocks.append(block)
                            captured_gb += (to_grab * 0.98)
                            print(f"  Secured chunk: {to_grab * 0.98:.2f} GB | Local Total: {captured_gb:.2f}/{min_memory_gb:.2f}")
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                
                time.sleep(0.5)

            # Inform reserver that we've secured what we need
            if captured_gb >= (min_memory_gb * 0.8) and signaled_pids:
                print(f"✓ Memory successfully consolidated ({captured_gb:.2f} GB). Sending Handover OK.")
                for pid in signaled_pids:
                    try:
                        os.kill(pid, signal.SIGUSR2) # SIGUSR2 = OK I got it
                    except ProcessLookupError: pass
            else:
                print(f"! Failed to consolidate full requirement. Got {captured_gb:.2f} GB. Moving forward anyway...")

            # Clean up our protective blocks so the main script can use the memory
            del captured_blocks
            torch.cuda.empty_cache()
            time.sleep(1.0) # Give nvidia-smi a bit more time to refresh
            
            # Final refresh
            gpu_info_final, _ = get_status()
            target_gpu = next((g for g in gpu_info_final if g[0] == best_gpu_idx), None)
            if target_gpu:
                # CRITICAL: Trust our own consolidation result over a potentially stale nvidia-smi poll.
                # If we successfully allocated 4.2GB in this process, we KNOW it is available now.
                best_real_free = max(target_gpu[1], captured_gb)
                best_effective_free = best_real_free 

    # 3. Final Selection
    # For decision making: if we didn't release, use best_effective_free to allow dashboard 
    # to show what is POSSIBLE with the reservation.
    actual_decision_VRAM = best_real_free if release_reserved else best_effective_free

    if best_gpu_idx is not None and actual_decision_VRAM >= min_memory_gb:
        # Initial device using global index
        device = torch.device(f'cuda:{best_gpu_idx}')
        
        # Set CUDA_VISIBLE_DEVICES to isolate the process to this GPU
        # Only do this if we are actually selecting for use (release_reserved=True)
        if release_reserved:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_idx)
            # After isolation, the device is always cuda:0
            device = torch.device('cuda:0')
            device_name = f"GPU {best_gpu_idx} ({best_gpu_name}) - {best_real_free:.2f} GB free"
        else:
            device_name = f"GPU {best_gpu_idx} ({best_gpu_name}) - {best_effective_free:.2f} GB effective (real: {best_real_free:.2f} GB)"
        
        print(f"\n✓ Selected: {device_name}")
        return device, device_name
    
    # Fallback to CPU
    # Use real free for the failure message if we are in active mode, otherwise effective for dashboard
    available_msg = best_real_free if release_reserved else best_effective_free
    device_name = f"CPU (GPU memory insufficient. Required: {min_memory_gb} GB, Best available: {available_msg:.2f} GB)"
    print(f"\n✓ Falling back to CPU: {device_name}")
    return torch.device('cpu'), device_name

def set_device(min_memory_gb=2.0):
    """
    Set and return the best device, also set CUDA memory growth to prevent OOM.
    """
    device, device_name = get_best_device(min_memory_gb, release_reserved=True)
    
    # Enable memory growth for CUDA to avoid pre-allocating all GPU memory
    if device.type == 'cuda':
        # Explicitly set the device in torch
        torch.cuda.set_device(device)
        # Clear cache to ensure we start fresh
        torch.cuda.empty_cache()
    
    return device, device_name

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU Utility and Reserver")
    parser.add_argument("--reserve", type=float, help="Reserve specific amount of GB on a GPU")
    parser.add_argument("--gpu_idx", type=int, help="Limit reservation to a specific GPU index (optional)")
    parser.add_argument("--min_gb", type=float, default=2.0, help="Min GB requirement for status check")
    
    args = parser.parse_args()
    
    if args.reserve:
        # gpu_idx is optional, if None it will poll ALL GPUs and take the first free one.
        reserve_gpu_memory(args.reserve, args.gpu_idx)
    else:
        # Dashbord View (No Arguments or status check)
        print("="*60)
        print("GPU STATUS DASHBOARD")
        print(f"Target Requirement: {args.min_gb} GB")
        print("="*60)
        
        # Don't release reservations during a simple status check
        device, device_name = get_best_device(min_memory_gb=args.min_gb, release_reserved=False)
        
        print("\nSUMMARY:")
        if device.type == "cuda":
            print(f"✅ READY: Will use {device_name}")
        else:
            print(f"❌ FALLBACK: {device_name}")
        
        print("\nRESERVATIONS:")
        import glob
        existing = glob.glob(get_lock_file())
        if existing:
            for f in existing:
                try:
                    with open(f, 'r') as rf:
                        data = rf.read().strip().split(':')
                        pid = data[0]
                        gpu_idx = data[1]
                        amount = data[2] if len(data) >= 3 else "Unknown"
                        print(f"  - Reservation: GPU {gpu_idx} | PID {pid} | Amount: {amount} GB")
                except:
                    pass
        else:
            print("  - No active reservations for current user.")
            
        print("="*60)
        print("Tip: Run 'python gpu_utils.py --reserve 8' to book a slot.")
        print("="*60)
