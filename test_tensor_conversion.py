#!/usr/bin/env python3
"""Test the tensor conversion fix"""
import torch

# Test the process_single_tensor function with different data types
def process_single_tensor(tensor, name):
    """Process a single tensor into the expected format"""
    # Convert Python lists to tensors first
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # Ensure tensor is on CPU
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    
    # Handle scalars (but not lists which were converted above)
    if not hasattr(tensor, 'shape'):
        return {
            'tensor': tensor,
            'shape': [],
            'dtype': str(type(tensor)),
            'mean': float(tensor),
            'std': 0.0,
            'min': float(tensor),
            'max': float(tensor),
        }
    
    # Handle regular tensors
    if tensor.numel() > 0:
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Replace NaN/Inf values with safe defaults
        mean_val = 0.0 if not torch.isfinite(torch.tensor(mean_val)) else mean_val
        std_val = 0.0 if not torch.isfinite(torch.tensor(std_val)) else std_val
        min_val = 0.0 if not torch.isfinite(torch.tensor(min_val)) else min_val
        max_val = 0.0 if not torch.isfinite(torch.tensor(max_val)) else max_val
    else:
        mean_val = std_val = min_val = max_val = 0.0
    
    return {
        'tensor': tensor,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
    }

# Test different input types
test_data = {
    'python_list': [1.0, 2.0, 3.0, 4.0],
    'tensor': torch.tensor([1.1, 2.1, 3.1, 4.1]),
    'scalar': 5.0
}

print("Testing tensor conversion:")
for name, data in test_data.items():
    print(f"\n{name}:")
    print(f"  Input: {data} (type: {type(data)})")
    result = process_single_tensor(data, name)
    print(f"  Output shape: {result['shape']}")
    print(f"  Output tensor: {result['tensor']}")
    print(f"  Has shape attr: {hasattr(result['tensor'], 'shape')}")