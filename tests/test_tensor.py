import torch

# Create a simple test tensor file with multiple tensors
test_data = {
    'tensor_a': torch.randn(2, 4, 8),
    'tensor_b': torch.randn(2, 4, 8),
    'tensor_c': torch.randn(16,),
    'scalar_d': torch.tensor(3.14),
}

torch.save(test_data, 'test_tensor.pth')
print("Created test_tensor.pth with 4 tensors for testing")
print("Tensor shapes:")
for name, tensor in test_data.items():
    if hasattr(tensor, 'shape'):
        print(f"  {name}: {tensor.shape}")
    else:
        print(f"  {name}: scalar ({tensor})")