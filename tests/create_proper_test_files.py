#!/usr/bin/env python3
"""Create proper .pth test files with actual tensor data"""

# Create test data that mimics what real .pth files contain
import pickle
import struct

# Create data that looks like serialized tensors
# This simulates what torch.save would create

def create_mock_tensor_data(values, shape):
    """Create mock tensor-like data structure"""
    return {
        'data': values,  # The actual values
        'shape': shape,  # Shape as a list  
        'dtype': 'float32',
        'device': 'cpu'
    }

# File 1 - Multi-dimensional tensors
test_data_1 = {
    'weight': create_mock_tensor_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]),
    'bias': create_mock_tensor_data([0.1, 0.2], [2])
}

# File 2 - Same structure, different values  
test_data_2 = {
    'weight': create_mock_tensor_data([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1], [2, 4]),
    'bias': create_mock_tensor_data([0.15, 0.25], [2])
}

# Save as pickle files with .pth extension
with open('test_tensor_1.pth', 'wb') as f:
    pickle.dump(test_data_1, f)

with open('test_tensor_2.pth', 'wb') as f:
    pickle.dump(test_data_2, f)

print("Created test_tensor_1.pth and test_tensor_2.pth")
print("File 1 tensors:", list(test_data_1.keys()))
print("File 2 tensors:", list(test_data_2.keys()))
print("Weight shape:", test_data_1['weight']['shape'])  
print("Bias shape:", test_data_1['bias']['shape'])