#!/usr/bin/env python3
"""Create real .pth test files with PyTorch tensors"""
import torch

# File 1 - Multi-dimensional tensors with real PyTorch tensors
test_data_1 = {
    'weight': torch.tensor([[1.0, 2.0, 3.0, 4.0], 
                           [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32),
    'bias': torch.tensor([0.1, 0.2], dtype=torch.float32)
}

# File 2 - Same structure, different values
test_data_2 = {
    'weight': torch.tensor([[1.1, 2.1, 3.1, 4.1], 
                           [5.1, 6.1, 7.1, 8.1]], dtype=torch.float32),
    'bias': torch.tensor([0.15, 0.25], dtype=torch.float32)
}

# Save as .pth files using torch.save
torch.save(test_data_1, 'real_tensor_1.pth')
torch.save(test_data_2, 'real_tensor_2.pth')

print("Created real_tensor_1.pth and real_tensor_2.pth")
print("File 1 tensors:", list(test_data_1.keys()))
print("File 2 tensors:", list(test_data_2.keys()))
print("Weight shape:", test_data_1['weight'].shape)  
print("Bias shape:", test_data_1['bias'].shape)
print("Weight values 1:", test_data_1['weight'])
print("Weight values 2:", test_data_2['weight'])