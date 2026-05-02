"""
Utility functions for reproducibility in PyTorch experiments.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Parameters:
    -----------
    seed : int
        Random seed value (default: 42)
    deterministic : bool
        If True, enable CUDA deterministic mode (may impact performance)
    
    Notes:
    ------
    - Sets seeds for Python random, NumPy, and PyTorch
    - Enables CUDA deterministic algorithms if available
    - For full reproducibility, also set:
      - torch.backends.cudnn.deterministic = True
      - torch.backends.cudnn.benchmark = False
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # CUDA deterministic mode (PyTorch >= 1.8)
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
            
            # Set environment variable for cuDNN
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_worker_init_fn(seed: int = 42):
    """
    Create a worker_init_fn for DataLoader to ensure reproducibility
    with multiple workers.
    
    Parameters:
    -----------
    seed : int
        Base random seed
    
    Returns:
    --------
    worker_init_fn : callable
        Function to initialize worker seeds
    """
    def worker_init_fn(worker_id):
        # Each worker gets a different seed based on worker_id
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn
