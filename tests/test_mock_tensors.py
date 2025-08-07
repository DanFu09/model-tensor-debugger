#!/usr/bin/env python3
"""Test loading the mock tensor files without PyTorch"""
import pickle

# Load and examine the test files 
print("Loading test_tensor_1.pth...")
with open('test_tensor_1.pth', 'rb') as f:
    data1 = pickle.load(f)

print("Loading test_tensor_2.pth...")  
with open('test_tensor_2.pth', 'rb') as f:
    data2 = pickle.load(f)

print("\nFile 1 contents:")
for key, value in data1.items():
    print(f"  {key}: {value}")

print("\nFile 2 contents:")
for key, value in data2.items():
    print(f"  {key}: {value}")

print("\nBoth files have keys:", list(data1.keys()))
print("Common keys:", set(data1.keys()) & set(data2.keys()))