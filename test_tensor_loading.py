#!/usr/bin/env python3

import os
import torch
import numpy as np
from pathlib import Path

def load_tensor_files(directory):
    """Load all .pth files from directory - refactored from app.py"""
    tensor_data = {}
    print(f"\n=== Loading tensors from: {directory} ===")
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory does not exist: {directory}")
        return tensor_data
    
    pth_files = list(Path(directory).rglob('*.pth'))
    print(f"Found {len(pth_files)} .pth files")
        
    for file_path in pth_files:
        try:
            print(f"Loading: {file_path.name}")
            # Try multiple loading strategies for compatibility
            tensor = None
            
            # Strategy 1: Normal loading
            try:
                tensor = torch.load(file_path, map_location='cpu')
            except Exception as e1:
                print(f"  Normal loading failed: {e1}")
                
                # Strategy 2: Load with weights_only=True (if available)
                try:
                    tensor = torch.load(file_path, map_location='cpu', weights_only=True)
                    print("  Loaded with weights_only=True")
                except Exception as e2:
                    print(f"  weights_only loading failed: {e2}")
                    
                    # Strategy 3: Try loading with pickle protocol
                    try:
                        import pickle
                        with open(file_path, 'rb') as f:
                            tensor = pickle.load(f)
                        print("  Loaded with pickle")
                    except Exception as e3:
                        print(f"  Pickle loading failed: {e3}")
                        print(f"  FAILED to load {file_path}")
                        continue
            
            if tensor is not None:
                rel_path = file_path.relative_to(directory)
                
                print(f"  ✓ Shape: {tensor.shape}, Dtype: {tensor.dtype}")
                
                tensor_data[str(rel_path)] = {
                    'tensor': tensor,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'mean': float(tensor.mean().item()) if tensor.numel() > 0 else 0,
                    'std': float(tensor.std().item()) if tensor.numel() > 0 else 0,
                    'min': float(tensor.min().item()) if tensor.numel() > 0 else 0,
                    'max': float(tensor.max().item()) if tensor.numel() > 0 else 0,
                }
                
        except Exception as e:
            print(f"  FINAL ERROR loading {file_path}: {e}")
    
    print(f"\nSuccessfully loaded {len(tensor_data)} tensors out of {len(pth_files)} total")
    return tensor_data

def parse_tensor_filename(filepath):
    """Parse tensor filename to extract layer and stage info"""
    base_name = Path(filepath).name
    parts = base_name.replace('.pth', '').split('_')
    
    try:
        layer_num = int(parts[0])
        stage = '_'.join(parts[1:])
        return layer_num, stage
    except (ValueError, IndexError) as e:
        print(f"Could not parse filename {filepath}: {e}")
        return None, None

def analyze_tensor_structure(tensor_data, model_name):
    """Analyze the structure of loaded tensors"""
    print(f"\n=== Analysis of {model_name} ===")
    
    # Group by layer
    layers = {}
    for filepath, data in tensor_data.items():
        layer_num, stage = parse_tensor_filename(filepath)
        if layer_num is not None:
            if layer_num not in layers:
                layers[layer_num] = {}
            layers[layer_num][stage] = filepath
            print(f"  {filepath} -> Layer {layer_num}, Stage '{stage}'")
        else:
            print(f"  UNPARSEABLE: {filepath}")
    
    print(f"\nLayer structure:")
    for layer_num in sorted(layers.keys()):
        print(f"  Layer {layer_num}: {sorted(layers[layer_num].keys())}")
    
    return layers

def compare_model_structures(model1_layers, model2_layers):
    """Compare the structures of two models"""
    print(f"\n=== Structure Comparison ===")
    
    all_layers = sorted(set(model1_layers.keys()) | set(model2_layers.keys()))
    stage_order = ['post_ln_pre_attn', 'post_attn', 'post_attn_pre_resid', 'pre_mlp', 'post_mlp']
    
    for layer_num in all_layers:
        print(f"\nLayer {layer_num}:")
        model1_stages = set(model1_layers.get(layer_num, {}).keys())
        model2_stages = set(model2_layers.get(layer_num, {}).keys())
        
        print(f"  Model 1 stages: {sorted(model1_stages)}")
        print(f"  Model 2 stages: {sorted(model2_stages)}")
        
        common_stages = model1_stages & model2_stages
        model1_only = model1_stages - model2_stages  
        model2_only = model2_stages - model1_stages
        
        print(f"  Common stages: {sorted(common_stages)}")
        if model1_only:
            print(f"  Model 1 only: {sorted(model1_only)}")
        if model2_only:
            print(f"  Model 2 only: {sorted(model2_only)}")
        
        print(f"  Expected stage matches:")
        for stage in stage_order:
            if stage in common_stages:
                print(f"    ✓ {stage}")
            elif stage in model1_stages and stage in model2_stages:
                print(f"    ✓ {stage} (both have it)")
            elif stage in model1_stages:
                print(f"    - {stage} (Model 1 only)")
            elif stage in model2_stages:
                print(f"    - {stage} (Model 2 only)")
            else:
                print(f"    ✗ {stage} (missing from both)")

if __name__ == "__main__":
    print("=== Tensor Loading Test Script ===")
    
    # Test directories
    model1_dir = os.path.expanduser("~/Downloads/outputs")
    model2_dir = os.path.expanduser("~/scp-files/gptoss/outputs_every_layer_test")
    
    print(f"Testing directories:")
    print(f"  Model 1: {model1_dir}")
    print(f"  Model 2: {model2_dir}")
    
    # Load tensors from both directories
    model1_data = load_tensor_files(model1_dir)
    model2_data = load_tensor_files(model2_dir)
    
    # Analyze structures
    model1_layers = analyze_tensor_structure(model1_data, "Model 1")
    model2_layers = analyze_tensor_structure(model2_data, "Model 2")
    
    # Compare structures
    compare_model_structures(model1_layers, model2_layers)
    
    print(f"\n=== Summary ===")
    print(f"Model 1 loaded {len(model1_data)} tensors")
    print(f"Model 2 loaded {len(model2_data)} tensors")