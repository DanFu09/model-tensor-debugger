#!/usr/bin/env python3
"""Create two simple .pth test files for dual comparison"""

# Create simple test data (no torch needed)
import pickle

# File 1 - Simple tensor-like structure
test_data_1 = {
    'weight': [1.0, 2.0, 3.0, 4.0],
    'bias': [0.1, 0.2]
}

# File 2 - Same structure, different values
test_data_2 = {
    'weight': [1.1, 2.1, 3.1, 4.1], 
    'bias': [0.15, 0.25]
}

# Save as pickle files with .pth extension (simulates torch.save format)
with open('test_file1.pth', 'wb') as f:
    pickle.dump(test_data_1, f)

with open('test_file2.pth', 'wb') as f:
    pickle.dump(test_data_2, f)

print("Created test_file1.pth and test_file2.pth")
print("File 1 tensors:", list(test_data_1.keys()))
print("File 2 tensors:", list(test_data_2.keys()))
print("Matching tensors should be: weight, bias")