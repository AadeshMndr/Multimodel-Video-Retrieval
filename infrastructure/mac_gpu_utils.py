"""
Mac GPU Optimization Utilities

This module provides utilities for optimizing performance on Apple Silicon Macs
with MPS (Metal Performance Shaders) support.
"""

import os
import logging
import torch

def setup_mac_gpu_environment():
    """
    Set up environment variables for optimal Mac GPU performance.
    Call this before initializing any GPU-intensive operations.
    """
    # Enable MPS fallback for unsupported operations
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    
    # Use available GPU memory efficiently
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
    
    logging.info("Mac GPU environment variables configured.")

def is_mac_gpu_available() -> bool:
    """Check if Apple Silicon GPU (MPS) is available"""
    try:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        return False

def clear_mac_gpu_cache():
    """Clear MPS memory cache to prevent memory issues"""
    if is_mac_gpu_available():
        try:
            torch.mps.empty_cache()
            import gc
            gc.collect()
            logging.debug("MPS cache cleared successfully")
        except Exception as e:
            logging.warning(f"Failed to clear MPS cache: {e}")

def get_recommended_batch_size(base_batch_size: int = 32) -> int:
    """
    Get recommended batch size based on available device.
    Doubles the batch size for GPU devices.
    """
    if torch.cuda.is_available() or is_mac_gpu_available():
        return base_batch_size * 2
    return base_batch_size

def print_device_info():
    """Print information about available devices"""
    print("=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS (Apple GPU) available: {is_mac_gpu_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    if is_mac_gpu_available():
        print("Apple Silicon GPU (MPS) detected")
    
    print(f"Default device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if is_mac_gpu_available() else 'cpu')}")
    print("=" * 26)