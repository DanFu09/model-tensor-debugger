#!/usr/bin/env python3
"""Direct test of dual .pth backend processing"""
import sys
import os
import torch
from pathlib import Path

# Add the parent directory to path so we can import app functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import process_single_tensor, DualPthStrategy, match_tensors, store_matches_for_inspection

def test_dual_pth_processing():
    """Test the dual .pth processing pipeline directly"""
    
    print("=== Testing Dual .pth Processing Pipeline ===")
    
    # Step 1: Load the files like the backend would
    print("\n1. Loading .pth files...")
    
    data1 = torch.load('real_tensor_1.pth', map_location='cpu')
    data2 = torch.load('real_tensor_2.pth', map_location='cpu')
    
    print(f"File 1 keys: {list(data1.keys())}")
    print(f"File 2 keys: {list(data2.keys())}")
    
    # Step 2: Process each tensor like handle_dual_pth_upload would
    print("\n2. Processing tensors...")
    
    model1_data = {}
    model2_data = {}
    
    # Process File 1
    for key, tensor in data1.items():
        model1_data[f"file1:{key}"] = process_single_tensor(tensor, key)
        print(f"File 1 - {key}: shape {model1_data[f'file1:{key}']['shape']}")
    
    # Process File 2  
    for key, tensor in data2.items():
        model2_data[f"file2:{key}"] = process_single_tensor(tensor, key)
        print(f"File 2 - {key}: shape {model2_data[f'file2:{key}']['shape']}")
    
    # Step 3: Match tensors using dual_pth strategy
    print("\n3. Matching tensors...")
    
    matches = match_tensors(model1_data, model2_data, strategy='dual_pth', upload_mode='dual_pth')
    
    print(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches):
        print(f"  Match {i}: {match['stage_display']}")
        print(f"    Model 1: {match['model1_data']['shape']}")
        print(f"    Model 2: {match['model2_data']['shape']}")
        if 'diff_stats' in match and match['diff_stats'] and not match['diff_stats'].get('error'):
            print(f"    Max diff: {match['diff_stats']['abs_diff_max']:.6f}")
            print(f"    Cosine sim: {match['diff_stats']['cosine_sim']:.4f}")
        else:
            print(f"    Error: {match['diff_stats'].get('error', 'Unknown')}")
    
    # Step 4: Store for inspection
    print("\n4. Storing matches for inspection...")
    
    stored_matches = store_matches_for_inspection(matches)
    
    print(f"Stored {len(stored_matches)} matches for inspection")
    for i, stored_match in enumerate(stored_matches):
        tensor1 = stored_match.get('tensor1')
        tensor2 = stored_match.get('tensor2')
        
        print(f"  Stored match {i}:")
        print(f"    Tensor1 shape: {tensor1.shape if hasattr(tensor1, 'shape') else 'No shape'}")
        print(f"    Tensor2 shape: {tensor2.shape if hasattr(tensor2, 'shape') else 'No shape'}")
    
    # Step 5: Test tensor value extraction (like the frontend would)
    print("\n5. Testing tensor value extraction...")
    
    if len(stored_matches) > 0:
        # Test with first match
        match = stored_matches[0]
        tensor1 = match['tensor1']
        tensor2 = match['tensor2']
        
        print(f"Testing match 0:")
        print(f"  Tensor1: {tensor1}")
        print(f"  Tensor2: {tensor2}")
        
        # Test slicing like the frontend would
        if hasattr(tensor1, 'shape') and len(tensor1.shape) > 0:
            # Try to get some values with dimension indices [0, 0] (if 2D)
            if len(tensor1.shape) >= 2:
                val1 = tensor1[0, 0].item()
                val2 = tensor2[0, 0].item()
                print(f"  Values at [0,0]: {val1} vs {val2}")
            elif len(tensor1.shape) == 1:
                val1 = tensor1[0].item()
                val2 = tensor2[0].item()
                print(f"  Values at [0]: {val1} vs {val2}")
    
    return matches, stored_matches

if __name__ == '__main__':
    try:
        matches, stored_matches = test_dual_pth_processing()
        print(f"\n=== Test completed successfully! ===")
        print(f"Matches: {len(matches)}, Stored: {len(stored_matches)}")
    except Exception as e:
        print(f"\n=== Test failed with error: ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()