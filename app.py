import os
import zipfile
import tarfile
import tempfile
import shutil
from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
import json
from pathlib import Path
import plotly.graph_objs as go
import plotly.utils

# Configure Flask to work with Vercel's directory structure
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size (reduced for serverless)

def extract_archive(file_path, extract_to):
    """Extract zip or tar.gz files"""
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError("Unsupported file format")

def load_tensor_files(directory):
    """Load all .pth files from directory with robust compatibility handling"""
    tensor_data = {}
    failed_files = []
    
    for file_path in Path(directory).rglob('*.pth'):
        try:
            print(f"Loading: {file_path.name}")
            tensor = None
            
            # Try multiple loading strategies for compatibility - ALWAYS map to CPU
            # Strategy 1: Normal loading
            try:
                tensor = torch.load(file_path, map_location='cpu')
                # Ensure tensor is definitely on CPU
                if hasattr(tensor, 'cpu'):
                    tensor = tensor.cpu()
            except Exception as e1:
                print(f"  Normal loading failed: {e1}")
                
                # Strategy 2: Load with weights_only=True (if available)
                try:
                    tensor = torch.load(file_path, map_location='cpu', weights_only=True)
                    # Ensure tensor is definitely on CPU
                    if hasattr(tensor, 'cpu'):
                        tensor = tensor.cpu()
                    print("  Loaded with weights_only=True")
                except Exception as e2:
                    print(f"  weights_only loading failed: {e2}")
                    failed_files.append(str(file_path))
                    continue
            
            if tensor is not None:
                rel_path = file_path.relative_to(directory)
                
                # Handle scalars (like sliding which is just a float/int)
                if hasattr(tensor, 'shape'):
                    print(f"  ✓ Shape: {tensor.shape}, Dtype: {tensor.dtype}")
                    
                    # Calculate statistics with NaN handling
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
                    
                    tensor_data[str(rel_path)] = {
                        'tensor': tensor,
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'min': float(min_val),
                        'max': float(max_val),
                    }
                else:
                    # Handle scalar values (like sliding)
                    print(f"  ✓ Scalar value: {tensor} (type: {type(tensor)})")
                    tensor_data[str(rel_path)] = {
                        'tensor': tensor,
                        'shape': [],  # Empty shape for scalar
                        'dtype': str(type(tensor)),
                        'mean': float(tensor),
                        'std': 0.0,
                        'min': float(tensor),
                        'max': float(tensor),
                    }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            failed_files.append(str(file_path))
    
    print(f"Successfully loaded {len(tensor_data)} tensors")
    if failed_files:
        print(f"Failed to load {len(failed_files)} tensors due to compatibility issues")
        print("Failed files (likely saved with newer PyTorch version):")
        for f in failed_files[:5]:  # Show first 5
            print(f"  - {Path(f).name}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    return tensor_data

class TensorMatchingStrategy:
    """Base class for tensor matching strategies"""
    def find_matches(self, model1_data, model2_data):
        raise NotImplementedError
    
    def get_strategy_name(self):
        return self.__class__.__name__

class LayerStageStrategy(TensorMatchingStrategy):
    """Original layer+stage matching logic"""
    def find_matches(self, model1_data, model2_data):
        return self._match_by_layer_stage(model1_data, model2_data)
    
    def _match_by_layer_stage(self, model1_data, model2_data):
        # This contains the existing match_tensors logic
        return match_tensors_layer_stage(model1_data, model2_data)

class ExactFilenameStrategy(TensorMatchingStrategy):
    """Match tensors by exact filename"""
    def find_matches(self, model1_data, model2_data):
        matches = []
        for file1, data1 in model1_data.items():
            filename1 = Path(file1).name
            for file2, data2 in model2_data.items():
                filename2 = Path(file2).name
                if filename1 == filename2:
                    # Create match
                    diff_stats = calculate_tensor_diff(data1['tensor'], data2['tensor'])
                    match = create_tensor_match(file1, data1, file2, data2, diff_stats, 'filename')
                    matches.append(match)
                    break
        return matches

class SingleFileStrategy(TensorMatchingStrategy):
    """Create pairwise comparisons within a single file"""
    def find_matches(self, model1_data, model2_data=None):
        # For single file mode, model2_data is ignored
        matches = []
        tensor_items = list(model1_data.items())
        
        for i, (name1, data1) in enumerate(tensor_items):
            for j, (name2, data2) in enumerate(tensor_items[i+1:], i+1):
                diff_stats = calculate_tensor_diff(data1['tensor'], data2['tensor'])
                match = create_tensor_match(name1, data1, name2, data2, diff_stats, 'self_comparison')
                match['match_index'] = len(matches)
                matches.append(match)
        
        return matches

class DualPthStrategy(TensorMatchingStrategy):
    """Match tensors from two .pth files by tensor names"""
    def find_matches(self, model1_data, model2_data):
        matches = []
        
        print(f"DEBUG: model1_data keys: {list(model1_data.keys())}")
        print(f"DEBUG: model2_data keys: {list(model2_data.keys())}")
        
        # Extract tensor names (after the file prefix)
        model1_tensors = {}
        model2_tensors = {}
        
        for key, data in model1_data.items():
            if ':' in key:
                tensor_name = key.split(':', 1)[1]  # Get part after 'file1:'
            else:
                # For single tensors, use generic name for comparison
                tensor_name = 'tensor' if key.startswith('file') else key
            model1_tensors[tensor_name] = (key, data)
        
        for key, data in model2_data.items():
            if ':' in key:
                tensor_name = key.split(':', 1)[1]  # Get part after 'file2:'
            else:
                # For single tensors, use generic name for comparison
                tensor_name = 'tensor' if key.startswith('file') else key
            model2_tensors[tensor_name] = (key, data)
        
        print(f"DEBUG: model1 tensor names: {list(model1_tensors.keys())}")
        print(f"DEBUG: model2 tensor names: {list(model2_tensors.keys())}")
        
        # Match by tensor name
        for tensor_name in model1_tensors:
            if tensor_name in model2_tensors:
                file1, data1 = model1_tensors[tensor_name]
                file2, data2 = model2_tensors[tensor_name]
                
                print(f"DEBUG: Matching tensor '{tensor_name}': {file1} vs {file2}")
                
                # Calculate difference statistics
                diff_stats = calculate_tensor_diff(data1['tensor'], data2['tensor'])
                match = create_tensor_match(file1, data1, file2, data2, diff_stats, 'dual_pth')
                match['match_index'] = len(matches)
                
                # Set layer info for dual .pth files (no actual layers, just direct comparison)
                match['layer_num'] = 0
                match['stage'] = tensor_name
                match['stage_index'] = len(matches)
                match['stage_display'] = tensor_name.replace('_', ' ').title()
                
                matches.append(match)
        
        print(f"DEBUG: Found {len(matches)} matches")
        return matches

def create_tensor_match(file1, data1, file2, data2, diff_stats, match_type='both'):
    """Create a standardized tensor match object"""
    return {
        'model1_file': file1,
        'model2_file': file2,
        'layer_num': 0,  # Default, can be overridden
        'stage': Path(file1).stem,  # Use filename as stage
        'stage_index': 0,
        'stage_display': Path(file1).stem.replace('_', ' ').title(),
        'model1_data': {k: v for k, v in data1.items() if k != 'tensor'},
        'model2_data': {k: v for k, v in data2.items() if k != 'tensor'},
        'diff_stats': diff_stats,
        'tensor1': data1['tensor'],
        'tensor2': data2['tensor'],
        'match_type': match_type
    }

def match_tensors_layer_stage(model1_data, model2_data):
    """Original layer+stage matching logic - moved from match_tensors"""
    # Group files by layer and stage
    model1_grouped = {}
    model2_grouped = {}
    
    # Define the expected stage order within each layer
    stage_order = [
        'post_ln_pre_attn', 
        'query', 'query_final',
        'key', 'key_final',
        'value', 'value_final',
        'post_attn', 
        'scaling', 'sliding',
        'post_attn_pre_resid', 
        'pre_mlp', 
        'post_mlp'
    ]
    
    # Debug: Print all found tensor files
    print("=== DEBUG: Found tensor files ===")
    print("Model 1 files:")
    for file, data in model1_data.items():
        print(f"  {file} -> shape: {data['shape']}")
    print("Model 2 files:")
    for file, data in model2_data.items():
        print(f"  {file} -> shape: {data['shape']}")
    print("=" * 40)
    
    # Parse model 1 files
    for file1, data1 in model1_data.items():
        base_name1 = Path(file1).name
        parts1 = base_name1.replace('.pth', '').split('_')
        try:
            layer_num1 = int(parts1[0])
            stage1 = '_'.join(parts1[1:])
            print(f"Model 1: {file1} -> layer {layer_num1}, stage '{stage1}'")
            
            if layer_num1 not in model1_grouped:
                model1_grouped[layer_num1] = {}
            model1_grouped[layer_num1][stage1] = {'file': file1, 'data': data1}
        except (ValueError, IndexError) as e:
            print(f"Could not parse Model 1 file {file1}: {e}")
    
    # Parse model 2 files
    for file2, data2 in model2_data.items():
        base_name2 = Path(file2).name
        parts2 = base_name2.replace('.pth', '').split('_')
        try:
            layer_num2 = int(parts2[0])
            stage2 = '_'.join(parts2[1:])
            print(f"Model 2: {file2} -> layer {layer_num2}, stage '{stage2}'")
            
            if layer_num2 not in model2_grouped:
                model2_grouped[layer_num2] = {}
            model2_grouped[layer_num2][stage2] = {'file': file2, 'data': data2}
        except (ValueError, IndexError) as e:
            print(f"Could not parse Model 2 file {file2}: {e}")
    
    # Match tensors by layer and stage order
    matches = []
    
    # Get all layers from both models and sort
    all_layers = sorted(set(model1_grouped.keys()) | set(model2_grouped.keys()))
    
    print(f"=== DEBUG: Matching process ===")
    print(f"Available layers: {all_layers}")
    print(f"Expected stage order: {stage_order}")
    
    for layer_num in all_layers:
        model1_layer_data = model1_grouped.get(layer_num, {})
        model2_layer_data = model2_grouped.get(layer_num, {})
        
        print(f"\nLayer {layer_num}:")
        print(f"  Model 1 stages: {list(model1_layer_data.keys())}")
        print(f"  Model 2 stages: {list(model2_layer_data.keys())}")
        
        # Show ALL stages from both models, even if not matching
        all_stages_this_layer = set(model1_layer_data.keys()) | set(model2_layer_data.keys())
        print(f"  All stages found: {sorted(all_stages_this_layer)}")
        
        # First, process stages in the defined order for both matched and unmatched tensors
        for stage_idx, stage in enumerate(stage_order):
            if stage in model1_layer_data and stage in model2_layer_data:
                print(f"    ✓ Matching stage: {stage}")
                data1 = model1_layer_data[stage]['data']
                data2 = model2_layer_data[stage]['data']
                file1 = model1_layer_data[stage]['file']
                file2 = model2_layer_data[stage]['file']
                
                # Calculate summary statistics
                diff_stats = calculate_tensor_diff(data1['tensor'], data2['tensor'])
                
                matches.append({
                    'model1_file': file1,
                    'model2_file': file2,
                    'layer_num': layer_num,
                    'stage': stage,
                    'stage_index': stage_idx,  # Order within the layer
                    'stage_display': stage.replace('_', ' ').title(),
                    'model1_data': {k: v for k, v in data1.items() if k != 'tensor'},
                    'model2_data': {k: v for k, v in data2.items() if k != 'tensor'},
                    'diff_stats': diff_stats,
                    'tensor1': data1['tensor'],  # Keep tensors for detailed inspection
                    'tensor2': data2['tensor'],
                    'match_type': 'both'
                })
            elif stage in model1_layer_data:
                print(f"    - Stage '{stage}' only in Model 1")
                data1 = model1_layer_data[stage]['data']
                file1 = model1_layer_data[stage]['file']
                
                matches.append({
                    'model1_file': file1,
                    'model2_file': None,
                    'layer_num': layer_num,
                    'stage': stage,
                    'stage_index': stage_idx,
                    'stage_display': stage.replace('_', ' ').title(),
                    'model1_data': {k: v for k, v in data1.items() if k != 'tensor'},
                    'model2_data': None,
                    'diff_stats': None,
                    'tensor1': data1['tensor'],
                    'tensor2': None,
                    'match_type': 'model1_only'
                })
            elif stage in model2_layer_data:
                print(f"    - Stage '{stage}' only in Model 2")
                data2 = model2_layer_data[stage]['data']
                file2 = model2_layer_data[stage]['file']
                
                matches.append({
                    'model1_file': None,
                    'model2_file': file2,
                    'layer_num': layer_num,
                    'stage': stage,
                    'stage_index': stage_idx,
                    'stage_display': stage.replace('_', ' ').title(),
                    'model1_data': None,
                    'model2_data': {k: v for k, v in data2.items() if k != 'tensor'},
                    'diff_stats': None,
                    'tensor1': None,
                    'tensor2': data2['tensor'],
                    'match_type': 'model2_only'
                })
            else:
                print(f"    - Stage '{stage}' missing from both models")
        
        # Now check for any additional stages not in our expected order
        matched_stages = set(stage for stage in stage_order 
                           if stage in model1_layer_data and stage in model2_layer_data)
        
        # Find stages that exist in both models but aren't in our predefined order
        all_common_stages = set(model1_layer_data.keys()) & set(model2_layer_data.keys())
        unexpected_stages = all_common_stages - matched_stages
        
        for stage in sorted(unexpected_stages):
            print(f"    ✓ Additional matching stage (not in expected order): {stage}")
            data1 = model1_layer_data[stage]['data']
            data2 = model2_layer_data[stage]['data']
            file1 = model1_layer_data[stage]['file']
            file2 = model2_layer_data[stage]['file']
            
            # Calculate summary statistics
            diff_stats = calculate_tensor_diff(data1['tensor'], data2['tensor'])
            
            matches.append({
                'model1_file': file1,
                'model2_file': file2,
                'layer_num': layer_num,
                'stage': stage,
                'stage_index': len(stage_order) + len([s for s in sorted(unexpected_stages) if s < stage]),  # Put after expected stages
                'stage_display': stage.replace('_', ' ').title(),
                'model1_data': {k: v for k, v in data1.items() if k != 'tensor'},
                'model2_data': {k: v for k, v in data2.items() if k != 'tensor'},
                'diff_stats': diff_stats,
                'tensor1': data1['tensor'],  # Keep tensors for detailed inspection
                'tensor2': data2['tensor']
            })
    
    # Sort matches by layer number, then by stage order
    matches.sort(key=lambda x: (x['layer_num'], x['stage_index']))
    
    return matches

def match_tensors(model1_data, model2_data, strategy='auto', upload_mode='dual_archive'):
    """Generalized tensor matching with multiple strategies"""
    
    # Handle single file mode
    if upload_mode == 'single_file':
        single_strategy = SingleFileStrategy()
        return single_strategy.find_matches(model1_data)
    
    # Handle dual .pth file mode
    if upload_mode == 'dual_pth':
        strategies = [DualPthStrategy()]  # Use specialized dual .pth matching
    else:
        # For dual archive mode, try multiple strategies
        strategies = []
        
        if strategy == 'auto' or strategy == 'layer_stage':
            strategies.append(LayerStageStrategy())
        if strategy == 'auto' or strategy == 'filename':
            strategies.append(ExactFilenameStrategy())
    
    # Try each strategy until we get matches
    for matching_strategy in strategies:
        try:
            matches = matching_strategy.find_matches(model1_data, model2_data)
            if matches:
                print(f"Successfully matched {len(matches)} tensor pairs using {matching_strategy.get_strategy_name()}")
                return matches
        except Exception as e:
            print(f"Strategy {matching_strategy.get_strategy_name()} failed: {e}")
            continue
    
    # If no strategy worked, return empty matches
    print("No matching strategy succeeded")
    return []

def can_reshape_tensors(shape1, shape2):
    """Check if tensors can be reshaped to match for different TP settings"""
    # If shapes are identical, no reshaping needed
    if shape1 == shape2:
        return True
    
    # Check if total elements match (basic requirement)
    total1 = np.prod(shape1) if len(shape1) > 0 else 1
    total2 = np.prod(shape2) if len(shape2) > 0 else 1
    if total1 != total2:
        return False
    
    # For tensor parallel differences, the last dimension should ideally match
    # or be compatible (e.g., one is a multiple of the other)
    if len(shape1) > 0 and len(shape2) > 0:
        last_dim1 = shape1[-1]
        last_dim2 = shape2[-1]
        
        # If last dimensions are the same, we can likely reshape
        if last_dim1 == last_dim2:
            return True
        
        # If one is a multiple of the other (common in TP settings), allow reshape
        if last_dim1 > 0 and last_dim2 > 0:
            if last_dim1 % last_dim2 == 0 or last_dim2 % last_dim1 == 0:
                return True
    
    # Otherwise, if total elements match, still allow reshape
    return True

def remove_leading_ones(shape):
    """Remove leading dimensions of size 1"""
    shape = list(shape)
    while len(shape) > 1 and shape[0] == 1:
        shape.pop(0)
    return shape

def find_transpose_match(shape1, shape2, preserve_last=True):
    """Find a transpose of two dimensions that makes shapes more compatible"""
    if len(shape1) != len(shape2) or len(shape1) < 2:
        return None, None
    
    # Remove leading 1s for analysis
    clean_shape1 = remove_leading_ones(shape1)
    clean_shape2 = remove_leading_ones(shape2)
    
    if len(clean_shape1) != len(clean_shape2) or len(clean_shape1) < 2:
        return None, None
    
    # Preserve last dimension if requested
    dims_to_try = list(range(len(clean_shape1) - (1 if preserve_last else 0)))
    
    # Try all possible transpositions of two dimensions
    for i in range(len(dims_to_try)):
        for j in range(i + 1, len(dims_to_try)):
            # Create transpose permutation
            perm = list(range(len(clean_shape1)))
            perm[i], perm[j] = perm[j], perm[i]
            
            # Check if transposing tensor1 makes it match tensor2
            transposed_shape1 = [clean_shape1[p] for p in perm]
            if transposed_shape1 == clean_shape2:
                # Map back to original tensor dimensions
                original_perm = list(range(len(shape1)))
                offset = len(shape1) - len(clean_shape1)
                if offset > 0:
                    # Adjust permutation for leading dimensions
                    full_perm = list(range(offset)) + [p + offset for p in perm]
                    return 1, full_perm
                else:
                    return 1, perm
            
            # Check if transposing tensor2 makes it match tensor1
            transposed_shape2 = [clean_shape2[p] for p in perm]
            if transposed_shape2 == clean_shape1:
                # Map back to original tensor dimensions
                original_perm = list(range(len(shape2)))
                offset = len(shape2) - len(clean_shape2)
                if offset > 0:
                    # Adjust permutation for leading dimensions
                    full_perm = list(range(offset)) + [p + offset for p in perm]
                    return 2, full_perm
                else:
                    return 2, perm
    
    return None, None

def apply_rank_zero_truncation(tensor1, tensor2):
    """Apply truncation assuming one tensor comes from rank zero (has extra data)"""
    shape1, shape2 = tensor1.shape, tensor2.shape
    
    # If same shape, no truncation needed
    if shape1 == shape2:
        return tensor1, tensor2
    
    # Check if one is a subset/truncation of the other
    # Strategy: find the smaller tensor and truncate the larger one to match
    total1, total2 = tensor1.numel(), tensor2.numel()
    
    if total1 == total2:
        # Same number of elements, try reshaping
        if len(shape1) <= len(shape2):
            try:
                tensor2_reshaped = tensor2.reshape(shape1)
                return tensor1, tensor2_reshaped
            except:
                pass
        if len(shape2) <= len(shape1):
            try:
                tensor1_reshaped = tensor1.reshape(shape2)
                return tensor1_reshaped, tensor2
            except:
                pass
    
    # Different number of elements - apply truncation
    if total1 < total2:
        # tensor1 is smaller, truncate tensor2
        try:
            # Try to truncate tensor2 to match tensor1's total elements
            tensor2_flat = tensor2.flatten()
            tensor2_truncated = tensor2_flat[:total1]
            tensor2_reshaped = tensor2_truncated.reshape(shape1)
            print(f"  Truncated tensor2 from {total2} to {total1} elements")
            return tensor1, tensor2_reshaped
        except Exception as e:
            print(f"  Failed to truncate tensor2: {e}")
    
    elif total2 < total1:
        # tensor2 is smaller, truncate tensor1
        try:
            # Try to truncate tensor1 to match tensor2's total elements
            tensor1_flat = tensor1.flatten()
            tensor1_truncated = tensor1_flat[:total2]
            tensor1_reshaped = tensor1_truncated.reshape(shape2)
            print(f"  Truncated tensor1 from {total1} to {total2} elements")
            return tensor1_reshaped, tensor2
        except Exception as e:
            print(f"  Failed to truncate tensor1: {e}")
    
    raise ValueError(f"Cannot apply rank zero truncation: {shape1} vs {shape2}")

def smart_reshape_for_tp(tensor1, tensor2):
    """Intelligently reshape tensors for tensor parallel compatibility"""
    # Ensure both tensors are on CPU
    tensor1 = tensor1.cpu() if hasattr(tensor1, 'cpu') else tensor1
    tensor2 = tensor2.cpu() if hasattr(tensor2, 'cpu') else tensor2
    
    shape1, shape2 = tensor1.shape, tensor2.shape
    
    # If already compatible, return as-is
    if shape1 == shape2:
        return tensor1, tensor2
    
    print(f"  Attempting TP reshaping: {list(shape1)} vs {list(shape2)}")
    
    # Step 1: Look for transpose of two dimensions that makes more dimensions match
    # Don't touch the last dimension, ignore leading dimensions of 1
    tensor_to_transpose, transpose_perm = find_transpose_match(shape1, shape2, preserve_last=True)
    
    if tensor_to_transpose is not None:
        try:
            if tensor_to_transpose == 1:
                tensor1_transposed = tensor1.permute(transpose_perm)
                print(f"  Transposed tensor1 with permutation {transpose_perm}: {list(shape1)} -> {list(tensor1_transposed.shape)}")
                if tensor1_transposed.shape == tensor2.shape:
                    return tensor1_transposed, tensor2
                else:
                    # Continue with truncation if shapes still don't match
                    tensor1, tensor2 = tensor1_transposed, tensor2
            elif tensor_to_transpose == 2:
                tensor2_transposed = tensor2.permute(transpose_perm)
                print(f"  Transposed tensor2 with permutation {transpose_perm}: {list(shape2)} -> {list(tensor2_transposed.shape)}")
                if tensor2_transposed.shape == tensor1.shape:
                    return tensor1, tensor2_transposed
                else:
                    # Continue with truncation if shapes still don't match
                    tensor1, tensor2 = tensor1, tensor2_transposed
        except Exception as e:
            print(f"  Transpose failed: {e}")
    
    # Step 2: Apply transformation like truncation (assume one comes from rank zero)
    try:
        return apply_rank_zero_truncation(tensor1, tensor2)
    except Exception as e:
        print(f"  Rank zero truncation failed: {e}")
    
    # Final fallback: flatten both tensors
    try:
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        min_size = min(tensor1_flat.numel(), tensor2_flat.numel())
        print(f"  Fallback: flattening and truncating to {min_size} elements")
        return tensor1_flat[:min_size], tensor2_flat[:min_size]
    except:
        raise ValueError(f"Cannot find compatible shapes for {shape1} and {shape2}")

def calculate_tensor_diff(tensor1, tensor2):
    """Calculate difference statistics between two tensors with smart TP reshaping"""
    try:
        # Ensure both tensors are on CPU
        tensor1 = tensor1.cpu() if hasattr(tensor1, 'cpu') else tensor1
        tensor2 = tensor2.cpu() if hasattr(tensor2, 'cpu') else tensor2
        
        original_shapes = (list(tensor1.shape), list(tensor2.shape))
        
        # Handle shape differences using smart TP-aware reshaping
        if tensor1.shape != tensor2.shape:
            if can_reshape_tensors(tensor1.shape, tensor2.shape):
                try:
                    tensor1, tensor2 = smart_reshape_for_tp(tensor1, tensor2)
                    print(f"  Successfully reshaped tensors from {original_shapes[0]} and {original_shapes[1]} to {list(tensor1.shape)} and {list(tensor2.shape)}")
                except Exception as reshape_error:
                    return {
                        'shape_match': False,
                        'shape1': original_shapes[0],
                        'shape2': original_shapes[1],
                        'error': f'TP-aware reshape failed: {str(reshape_error)}',
                        'reshapeable': True
                    }
            else:
                return {
                    'shape_match': False,
                    'shape1': original_shapes[0],
                    'shape2': original_shapes[1],
                    'error': 'Incompatible tensor sizes for TP reshaping',
                    'reshapeable': False
                }
        
        # Verify shapes match after potential reshaping
        if tensor1.shape != tensor2.shape:
            return {
                'shape_match': False,
                'shape1': original_shapes[0],
                'shape2': original_shapes[1],
                'error': f'Shape mismatch after reshaping: {tensor1.shape} vs {tensor2.shape}',
                'reshapeable': False
            }
        
        diff = tensor1 - tensor2
        
        # Calculate statistics with NaN handling
        abs_diff_mean = torch.abs(diff).mean().item()
        abs_diff_max = torch.abs(diff).max().item()
        rel_diff_mean = (torch.abs(diff) / (torch.abs(tensor1) + 1e-8)).mean().item()
        mse = torch.mean(diff**2).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            tensor1.flatten(), tensor2.flatten(), dim=0).item()
        
        # Replace NaN values with 0.0
        abs_diff_mean = 0.0 if torch.isnan(torch.tensor(abs_diff_mean)) else abs_diff_mean
        abs_diff_max = 0.0 if torch.isnan(torch.tensor(abs_diff_max)) else abs_diff_max
        rel_diff_mean = 0.0 if torch.isnan(torch.tensor(rel_diff_mean)) else rel_diff_mean
        mse = 0.0 if torch.isnan(torch.tensor(mse)) else mse
        cosine_sim = 1.0 if torch.isnan(torch.tensor(cosine_sim)) else cosine_sim  # Perfect similarity if NaN
        
        return {
            'shape_match': True,
            'original_shape1': original_shapes[0],
            'original_shape2': original_shapes[1],
            'final_shape': list(tensor1.shape),
            'reshaped': original_shapes[0] != list(tensor1.shape) or original_shapes[1] != list(tensor2.shape),
            'abs_diff_mean': float(abs_diff_mean),
            'abs_diff_max': float(abs_diff_max),
            'rel_diff_mean': float(rel_diff_mean),
            'mse': float(mse),
            'cosine_sim': float(cosine_sim),
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Model Tensor Debugger',
        'version': '1.0.0'
    })

def detect_upload_mode(files, form_data):
    """Detect upload mode based on provided files"""
    if 'upload_mode' in form_data:
        return form_data['upload_mode']
    
    # Auto-detect based on files present
    if 'single_file' in files:
        return 'single_file'
    elif 'model1' in files and 'model2' in files:
        # Check if files are .pth files for direct comparison
        model1_name = files['model1'].filename.lower()
        model2_name = files['model2'].filename.lower()
        
        if (model1_name.endswith(('.pth', '.pt')) and 
            model2_name.endswith(('.pth', '.pt'))):
            return 'dual_pth'
        else:
            return 'dual_archive'
    else:
        return 'unknown'

def handle_single_file_upload(files):
    """Handle upload of single .pth file for self-comparison"""
    single_file = files['single_file']
    
    if single_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, single_file.filename)
        single_file.save(file_path)
        
        # Load the tensor file directly
        print(f"Loading single file: {single_file.filename}")
        tensor_data = {}
        
        try:
            # Try multiple loading strategies for compatibility
            data = None
            try:
                data = torch.load(file_path, map_location='cpu')
            except Exception as e1:
                print(f"  Normal loading failed: {e1}")
                try:
                    data = torch.load(file_path, map_location='cpu', weights_only=True)
                    print("  Loaded with weights_only=True")
                except Exception as e2:
                    return jsonify({'error': f'Failed to load tensor file with both loading methods. Normal: {str(e1)}, weights_only: {str(e2)}'}), 400
            
            if data is None:
                return jsonify({'error': 'Failed to load tensor data from file'}), 400
                
            if isinstance(data, dict):
                # File contains multiple tensors as dictionary
                if len(data) == 0:
                    return jsonify({'error': 'Tensor file contains empty dictionary'}), 400
                for key, tensor in data.items():
                    tensor_data[key] = process_single_tensor(tensor, key)
            elif isinstance(data, (list, tuple)):
                # File contains list/tuple of tensors
                if len(data) == 0:
                    return jsonify({'error': 'Tensor file contains empty list/tuple'}), 400
                for i, tensor in enumerate(data):
                    tensor_data[f'tensor_{i}'] = process_single_tensor(tensor, f'tensor_{i}')
            else:
                # File contains single tensor
                tensor_data['tensor_0'] = process_single_tensor(data, 'single_tensor')
                
        except Exception as e:
            return jsonify({'error': f'Failed to process tensor file: {str(e)}'}), 400
        
        # Create self-comparison matches
        matches = match_tensors(tensor_data, None, strategy='auto', upload_mode='single_file')
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'model1_files': len(tensor_data),
            'model2_files': len(tensor_data), 
            'matches': [serialize_match_for_response(match) for match in matches],
            'upload_mode': 'single_file'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def sanitize_for_json(tensor_values):
    """Convert tensor values to JSON-safe format, handling NaN and Inf"""
    if tensor_values is None:
        return None
    
    def clean_value(val):
        if isinstance(val, (list, tuple)):
            return [clean_value(v) for v in val]
        elif isinstance(val, float):
            if val != val:  # NaN check
                return 0.0
            elif val == float('inf'):
                return 1e10  # Large but finite number
            elif val == float('-inf'):
                return -1e10
            else:
                return val
        else:
            return val
    
    return clean_value(tensor_values)

def process_single_tensor(tensor, name):
    """Process a single tensor into the expected format"""
    # Handle mock tensor data structure (for testing without PyTorch)
    if isinstance(tensor, dict) and 'data' in tensor and 'shape' in tensor:
        # Convert mock tensor to actual tensor
        data = tensor['data']
        shape = tensor['shape']
        if isinstance(data, list):
            tensor = torch.tensor(data, dtype=torch.float32).reshape(shape)
    
    # Convert Python lists to tensors first
    elif isinstance(tensor, list):
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
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(min_val),
        'max': float(max_val),
    }

def serialize_match_for_response(match):
    """Remove tensors from match for JSON response"""
    match_copy = match.copy()
    match_copy.pop('tensor1', None)
    match_copy.pop('tensor2', None)
    return match_copy

@app.route('/upload', methods=['POST'])
def upload_files():
    upload_mode = detect_upload_mode(request.files, request.form)
    
    if upload_mode == 'single_file':
        return handle_single_file_upload(request.files)
    elif upload_mode == 'dual_archive':
        return handle_dual_archive_upload(request.files)
    elif upload_mode == 'dual_pth':
        return handle_dual_pth_upload(request.files)
    else:
        return jsonify({'error': 'Invalid upload mode or missing files'}), 400

def handle_dual_pth_upload(files):
    """Handle direct comparison of two .pth files"""
    if 'model1' not in files or 'model2' not in files:
        return jsonify({'error': 'Both tensor files required'}), 400
    
    model1_file = files['model1']
    model2_file = files['model2']
    
    if model1_file.filename == '' or model2_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save files
        model1_path = os.path.join(temp_dir, model1_file.filename)
        model2_path = os.path.join(temp_dir, model2_file.filename)
        
        model1_file.save(model1_path)
        model2_file.save(model2_path)
        
        print(f"Loading tensor files: {model1_file.filename} and {model2_file.filename}")
        
        # Load tensor data directly (not from archives)
        model1_data = {}
        model2_data = {}
        
        # Load first file
        try:
            data1 = torch.load(model1_path, map_location='cpu')
            if isinstance(data1, dict):
                for key, tensor in data1.items():
                    model1_data[f"file1:{key}"] = process_single_tensor(tensor, key)
            else:
                model1_data["file1"] = process_single_tensor(data1, 'tensor')
        except Exception as e:
            return jsonify({'error': f'Failed to load File 1: {str(e)}'}), 400
        
        # Load second file  
        try:
            data2 = torch.load(model2_path, map_location='cpu')
            if isinstance(data2, dict):
                for key, tensor in data2.items():
                    model2_data[f"file2:{key}"] = process_single_tensor(tensor, key)
            else:
                model2_data["file2"] = process_single_tensor(data2, 'tensor')
        except Exception as e:
            return jsonify({'error': f'Failed to load File 2: {str(e)}'}), 400
        
        # Match tensors using dual .pth strategy
        matches = match_tensors(model1_data, model2_data, strategy='dual_pth', upload_mode='dual_pth')
        
        print(f"DEBUG: dual_pth matches found: {len(matches)}")
        for i, match in enumerate(matches):
            print(f"  Match {i}: {match.get('model1_file', 'N/A')} vs {match.get('model2_file', 'N/A')}")
        
        # Store matches globally for tensor inspection
        global stored_matches
        stored_matches = store_matches_for_inspection(matches)
        
        print(f"DEBUG: stored_matches for dual_pth: {len(stored_matches)} matches")
        for i, stored_match in enumerate(stored_matches):
            print(f"  Match {i}: tensor1 shape: {stored_match.get('tensor1', 'None')}")
            print(f"  Match {i}: tensor2 shape: {stored_match.get('tensor2', 'None')}")
            if hasattr(stored_match.get('tensor1'), 'shape'):
                print(f"  Match {i}: tensor1 actual shape: {stored_match['tensor1'].shape}")
            if hasattr(stored_match.get('tensor2'), 'shape'):
                print(f"  Match {i}: tensor2 actual shape: {stored_match['tensor2'].shape}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'model1_files': len(model1_data),
            'model2_files': len(model2_data),
            'matches': [serialize_match_for_response(match) for match in matches],
            'upload_mode': 'dual_pth'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_dual_archive_upload(files):
    """Handle the original dual archive upload mode"""
    if 'model1' not in files or 'model2' not in files:
        return jsonify({'error': 'Both model files required'}), 400
    
    model1_file = files['model1']
    model2_file = files['model2']
    
    if model1_file.filename == '' or model2_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        model1_dir = os.path.join(temp_dir, 'model1')
        model2_dir = os.path.join(temp_dir, 'model2')
        os.makedirs(model1_dir)
        os.makedirs(model2_dir)
        
        # Save and extract files
        model1_path = os.path.join(temp_dir, model1_file.filename)
        model2_path = os.path.join(temp_dir, model2_file.filename)
        
        model1_file.save(model1_path)
        model2_file.save(model2_path)
        
        extract_archive(model1_path, model1_dir)
        extract_archive(model2_path, model2_dir)
        
        # Load tensor data
        model1_data = load_tensor_files(model1_dir)
        model2_data = load_tensor_files(model2_dir)
        
        # Match tensors using generalized matching system
        matches = match_tensors(model1_data, model2_data, strategy='auto', upload_mode='dual_archive')
        
        # Store matches globally for tensor inspection with both original and reshaped tensors
        global stored_matches
        stored_matches = store_matches_for_inspection(matches)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'model1_files': len(model1_data),
            'model2_files': len(model2_data),
            'matches': [serialize_match_for_response(match) for match in matches],
            'upload_mode': 'dual_archive'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def store_matches_for_inspection(matches):
    """Store matches with original and reshaped tensors for inspection"""
    stored = []
    
    for match in matches:
        # Create a stored match with both original and potentially reshaped tensors
        stored_match = match.copy()
        
        if match['tensor1'] is not None and match['tensor2'] is not None:
            # Both tensors available - check if reshaping is needed
            tensor1 = match['tensor1'].cpu() if hasattr(match['tensor1'], 'cpu') else match['tensor1']
            tensor2 = match['tensor2'].cpu() if hasattr(match['tensor2'], 'cpu') else match['tensor2']
            
            # Handle scalar values - they don't have shapes
            has_shape1 = hasattr(tensor1, 'shape')
            has_shape2 = hasattr(tensor2, 'shape')
            
            if not has_shape1 or not has_shape2:
                # At least one is a scalar - no reshaping needed
                original_shape1 = list(tensor1.shape) if has_shape1 else []
                original_shape2 = list(tensor2.shape) if has_shape2 else []
                
                stored_match.update({
                    'original_tensor1': tensor1,
                    'original_tensor2': tensor2,
                    'tensor1': tensor1,
                    'tensor2': tensor2,
                    'reshape_applied': False,
                    'original_shapes': [original_shape1, original_shape2],
                    'final_shape': original_shape1 if has_shape1 else original_shape2
                })
            else:
                # Both have shapes - proceed with normal tensor processing
                original_shape1 = list(tensor1.shape)
                original_shape2 = list(tensor2.shape)
            
                if tensor1.shape != tensor2.shape:
                    try:
                        # Apply TP-aware reshaping during upload phase
                        reshaped_tensor1, reshaped_tensor2 = smart_reshape_for_tp(tensor1, tensor2)
                        print(f"  Stored reshaped tensors for {match['model1_file']} and {match['model2_file']}")
                        print(f"  Original shapes: {original_shape1} vs {original_shape2} → Final: {list(reshaped_tensor1.shape)}")
                        
                        stored_match.update({
                            'original_tensor1': tensor1,
                            'original_tensor2': tensor2,
                            'tensor1': reshaped_tensor1,  # Use reshaped versions for slicing
                            'tensor2': reshaped_tensor2,
                            'reshape_applied': True,
                            'original_shapes': [original_shape1, original_shape2],
                            'final_shape': list(reshaped_tensor1.shape)
                        })
                    except Exception as e:
                        print(f"  Could not reshape tensors during upload for {match['model1_file']}: {e}")
                        stored_match.update({
                            'original_tensor1': tensor1,
                            'original_tensor2': tensor2,
                            'tensor1': tensor1,
                            'tensor2': tensor2,
                            'reshape_applied': False,
                            'original_shapes': [original_shape1, original_shape2],
                            'final_shape': None
                        })
                else:
                    # Shapes already match - no reshaping needed
                    stored_match.update({
                        'original_tensor1': tensor1,
                        'original_tensor2': tensor2,
                        'tensor1': tensor1,
                        'tensor2': tensor2,
                        'reshape_applied': False,
                        'original_shapes': [original_shape1, original_shape2],
                        'final_shape': list(tensor1.shape)
                    })
        else:
            # Only one tensor available - no reshaping needed
            if match['tensor1'] is not None:
                tensor1 = match['tensor1'].cpu() if hasattr(match['tensor1'], 'cpu') else match['tensor1']
                shape1 = list(tensor1.shape) if hasattr(tensor1, 'shape') else []
                stored_match.update({
                    'original_tensor1': tensor1,
                    'original_tensor2': None,
                    'tensor1': tensor1,
                    'tensor2': None,
                    'reshape_applied': False,
                    'original_shapes': [shape1, None],
                    'final_shape': shape1
                })
            elif match['tensor2'] is not None:
                tensor2 = match['tensor2'].cpu() if hasattr(match['tensor2'], 'cpu') else match['tensor2']
                shape2 = list(tensor2.shape) if hasattr(tensor2, 'shape') else []
                stored_match.update({
                    'original_tensor1': None,
                    'original_tensor2': tensor2,
                    'tensor1': None,
                    'tensor2': tensor2,
                    'reshape_applied': False,
                    'original_shapes': [None, shape2],
                    'final_shape': shape2
                })
            
        stored.append(stored_match)
    
    return stored


# Global storage for tensor data (in production, use proper session management)
stored_matches = []

@app.route('/get_tensor_values', methods=['POST'])
def get_tensor_values():
    data = request.json
    match_index = data.get('match_index', 0)
    dimension_indices = data.get('dimension_indices', [])
    count = data.get('count', 10)
    get_argmax = data.get('get_argmax', False)
    get_final_shape = data.get('get_final_shape', False)
    
    if match_index >= len(stored_matches):
        return jsonify({'error': 'Invalid match index'}), 400
    
    match = stored_matches[match_index]
    tensor1 = match['tensor1']
    tensor2 = match['tensor2']
    
    try:
        # Determine which tensors are available
        has_tensor1 = tensor1 is not None
        has_tensor2 = tensor2 is not None
        
        # Get reshape information from stored match
        original_shapes = match.get('original_shapes')
        reshape_applied = match.get('reshape_applied', False)
        final_shape = match.get('final_shape')
        
        # Use the pre-reshaped tensors stored during upload phase
        # No additional reshaping is done here - tensors should already be compatible
        
        # Ensure tensors are on CPU (they should already be from upload phase)
        if has_tensor1 and hasattr(tensor1, 'cpu'):
            tensor1 = tensor1.cpu()
        if has_tensor2 and hasattr(tensor2, 'cpu'):
            tensor2 = tensor2.cpu()
        
        if reshape_applied and original_shapes:
            print(f"  Using pre-reshaped tensors for {match.get('model1_file', 'tensor1')} and {match.get('model2_file', 'tensor2')}")
            print(f"  Original shapes: {original_shapes[0]} vs {original_shapes[1]} → Final: {final_shape}")
            # Convert original_shapes to the format expected by the rest of the function
            original_shapes = {
                'shape1': original_shapes[0] if original_shapes[0] else [],
                'shape2': original_shapes[1] if original_shapes[1] else []
            }
        
        # Use the available tensor to determine shape and processing
        primary_tensor = tensor1 if has_tensor1 else tensor2
        
        # If just requesting final shape info, return it early
        if get_final_shape:
            return jsonify({
                'tensor_shape': list(primary_tensor.shape),
                'original_shapes': original_shapes,
                'display_type': f'{len(primary_tensor.shape)}D tensor',
                'slice_info': 'Shape information only'
            })
        
        # Handle scalars
        if not hasattr(primary_tensor, 'shape') or len(primary_tensor.shape) == 0:
            # Handle scalar values
            values1 = [float(tensor1)] if has_tensor1 else None
            values2 = [float(tensor2)] if has_tensor2 else None
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'tensor_shape': [],
                'display_type': 'scalar',
                'argmax_index': None,
                'argmax_coordinates': None,
                'slice_info': 'Scalar value'
            })
        
        # Calculate argmax coordinates if requested and both tensors available
        argmax_coords = None
        if get_argmax and has_tensor1 and has_tensor2:
            try:
                abs_diff = torch.abs(tensor1 - tensor2)
                flat_argmax_idx = int(torch.argmax(abs_diff).item())
                # Convert flat index to multi-dimensional coordinates
                argmax_coords = [int(x) for x in torch.unravel_index(torch.tensor(flat_argmax_idx), tensor1.shape)]
                print(f"  Argmax coordinates: {argmax_coords} for tensor shape {list(tensor1.shape)}")
            except Exception as e:
                print(f"  Could not calculate argmax coordinates: {e}")
        
        # Handle dimension slicing based on tensor dimensionality
        if len(primary_tensor.shape) == 1:
            # 1D tensor - use dimension index directly
            dim_idx = dimension_indices[0] if dimension_indices else 0
            start_idx = max(0, dim_idx)
            end_idx = min(start_idx + count, primary_tensor.shape[0])
            
            values1 = sanitize_for_json(tensor1[start_idx:end_idx].tolist()) if has_tensor1 else None
            values2 = sanitize_for_json(tensor2[start_idx:end_idx].tolist()) if has_tensor2 else None
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'tensor_shape': list(primary_tensor.shape),
                'display_type': '1D tensor',
                'argmax_index': argmax_coords[0] if argmax_coords else None,
                'argmax_coordinates': argmax_coords,
                'slice_info': f'Values {start_idx} to {end_idx-1}'
            })
        
        elif len(primary_tensor.shape) >= 2:
            # Multi-dimensional tensor - create slices based on dimension indices
            tensor_shape = list(primary_tensor.shape)
            
            # Use the final tensor shape after any TP-aware reshaping for slicing
            # The dimension_indices should match the final tensor shape, not the original
            actual_shape = tensor_shape
            
            # Adjust dimension_indices for the actual tensor shape after reshaping
            if len(dimension_indices) > len(actual_shape):
                # Truncate if too many indices provided
                dim_indices = dimension_indices[:len(actual_shape)]
            else:
                # Pad with zeros if not enough indices
                dim_indices = dimension_indices + [0] * (len(actual_shape) - len(dimension_indices))
            
            # Create slice objects for each dimension
            slices = []
            slice_info_parts = []
            
            for i, (dim_size, idx) in enumerate(zip(actual_shape, dim_indices)):
                idx = max(0, min(idx, dim_size - 1))  # Clamp to valid range
                
                if i == len(actual_shape) - 1:  # Last dimension - slice a range of values
                    start_idx = idx
                    end_idx = min(start_idx + count, dim_size)
                    slices.append(slice(start_idx, end_idx))
                    slice_info_parts.append(f'dim{i}[{start_idx}:{end_idx}]')
                else:  # Other dimensions - use specific index
                    slices.append(idx)
                    slice_info_parts.append(f'dim{i}[{idx}]')
            
            # Extract the sliced values
            try:
                slice_tuple = tuple(slices)
                sliced1 = tensor1[slice_tuple] if has_tensor1 else None
                sliced2 = tensor2[slice_tuple] if has_tensor2 else None
                
                # Convert to list format
                values1 = sanitize_for_json(sliced1.tolist()) if has_tensor1 and sliced1 is not None else None
                values2 = sanitize_for_json(sliced2.tolist()) if has_tensor2 and sliced2 is not None else None
                
                # Handle the case where we get a single value vs a list
                if isinstance(values1, (int, float)):
                    values1 = [values1]
                if isinstance(values2, (int, float)):
                    values2 = [values2]
                
                slice_info = ' × '.join(slice_info_parts)
                display_type = f'{len(actual_shape)}D tensor slice'
                
                return jsonify({
                    'tensor1_values': values1,
                    'tensor2_values': values2,
                    'tensor_shape': actual_shape,
                    'display_type': display_type,
                    'argmax_index': None,  # Not applicable for sliced view
                    'argmax_coordinates': argmax_coords,
                    'slice_info': slice_info
                })
                
            except Exception as slice_error:
                # Fallback to flattened view
                print(f"  Slice failed, falling back to flattened view: {slice_error}")
                flat1 = tensor1.flatten() if has_tensor1 else None
                flat2 = tensor2.flatten() if has_tensor2 else None
                
                start_idx = 0
                end_idx = min(count, flat1.shape[0] if flat1 is not None else flat2.shape[0])
                
                values1 = sanitize_for_json(flat1[start_idx:end_idx].tolist()) if flat1 is not None else None
                values2 = sanitize_for_json(flat2[start_idx:end_idx].tolist()) if flat2 is not None else None
                
                return jsonify({
                    'tensor1_values': values1,
                    'tensor2_values': values2,
                    'tensor_shape': tensor_shape,
                    'display_type': 'flattened (slice failed)',
                    'argmax_index': None,
                    'argmax_coordinates': argmax_coords,
                    'slice_info': f'Flattened view: elements 0-{end_idx-1}'
                })
        
    except Exception as e:
        return jsonify({'error': f'Error extracting tensor values: {str(e)}'}), 400

@app.route('/get_tensor_shapes', methods=['POST'])
def get_tensor_shapes():
    """Get original tensor shapes before any reshaping for manual dimension mapping"""
    data = request.json
    match_index = data.get('match_index', 0)
    
    if match_index >= len(stored_matches):
        return jsonify({'error': 'Invalid match index'}), 400
    
    match = stored_matches[match_index]
    
    try:
        result = {}
        
        # Get original shapes from stored match metadata
        original_shapes = match.get('original_shapes', [None, None])
        
        if match.get('tensor1') is not None:
            # Use original tensor shape, not reshaped
            original_tensor1 = match.get('original_tensor1', match['tensor1'])
            if hasattr(original_tensor1, 'shape'):
                result['tensor1_shape'] = list(original_tensor1.shape)
                result['tensor1_name'] = match.get('model1_file', 'Model 1')
            else:
                result['tensor1_shape'] = []  # Scalar
                result['tensor1_name'] = match.get('model1_file', 'Model 1 (Scalar)')
                
        if match.get('tensor2') is not None:
            # Use original tensor shape, not reshaped
            original_tensor2 = match.get('original_tensor2', match['tensor2'])
            if hasattr(original_tensor2, 'shape'):
                result['tensor2_shape'] = list(original_tensor2.shape)
                result['tensor2_name'] = match.get('model2_file', 'Model 2')
            else:
                result['tensor2_shape'] = []  # Scalar
                result['tensor2_name'] = match.get('model2_file', 'Model 2 (Scalar)')
                
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error getting tensor shapes: {str(e)}'}), 400

@app.route('/get_tensor_values_manual', methods=['POST'])
def get_tensor_values_manual():
    """Get tensor values using manual dimension mapping"""
    data = request.json
    match_index = data.get('match_index', 0)
    tensor1_indices = data.get('tensor1_indices', [])
    tensor2_indices = data.get('tensor2_indices', [])
    count = data.get('count', 10)
    
    if match_index >= len(stored_matches):
        return jsonify({'error': 'Invalid match index'}), 400
    
    match = stored_matches[match_index]
    
    try:
        # Use ORIGINAL tensors before any reshaping
        tensor1 = match.get('original_tensor1')
        tensor2 = match.get('original_tensor2')
        
        result = {}
        sliced_tensor1 = None
        sliced_tensor2 = None
        
        # Process tensor 1
        if tensor1 is not None:
            if hasattr(tensor1, 'shape') and len(tensor1.shape) > 0:
                # Apply manual slicing to tensor 1
                sliced_tensor1, slice_info1 = slice_tensor_manually(tensor1, tensor1_indices, count)
                result['tensor1_values'] = sanitize_for_json(sliced_tensor1.tolist())
                result['tensor1_slice_info'] = slice_info1
            else:
                # Scalar tensor
                result['tensor1_values'] = [float(tensor1)]
                result['tensor1_slice_info'] = 'scalar'
                sliced_tensor1 = torch.tensor([float(tensor1)])
        
        # Process tensor 2
        if tensor2 is not None:
            if hasattr(tensor2, 'shape') and len(tensor2.shape) > 0:
                # Apply manual slicing to tensor 2
                sliced_tensor2, slice_info2 = slice_tensor_manually(tensor2, tensor2_indices, count)
                result['tensor2_values'] = sanitize_for_json(sliced_tensor2.tolist())
                result['tensor2_slice_info'] = slice_info2
            else:
                # Scalar tensor
                result['tensor2_values'] = [float(tensor2)]
                result['tensor2_slice_info'] = 'scalar'
                sliced_tensor2 = torch.tensor([float(tensor2)])
        
        # Calculate summary statistics for the sliced tensors
        if sliced_tensor1 is not None and sliced_tensor2 is not None:
            try:
                # Ensure both slices are tensors and have the same shape
                if not isinstance(sliced_tensor1, torch.Tensor):
                    sliced_tensor1 = torch.tensor(sliced_tensor1)
                if not isinstance(sliced_tensor2, torch.Tensor):
                    sliced_tensor2 = torch.tensor(sliced_tensor2)
                
                # Flatten both to 1D for comparison if they have different shapes
                if sliced_tensor1.shape != sliced_tensor2.shape:
                    min_elements = min(sliced_tensor1.numel(), sliced_tensor2.numel())
                    sliced_tensor1 = sliced_tensor1.flatten()[:min_elements]
                    sliced_tensor2 = sliced_tensor2.flatten()[:min_elements]
                
                # Calculate difference statistics
                diff = sliced_tensor1 - sliced_tensor2
                abs_diff = torch.abs(diff)
                
                # Calculate statistics with NaN handling
                abs_diff_mean = abs_diff.mean().item()
                abs_diff_max = abs_diff.max().item()
                rel_diff_mean = (abs_diff / (torch.abs(sliced_tensor1) + 1e-8)).mean().item()
                mse = torch.mean(diff**2).item()
                
                # Calculate cosine similarity if both tensors have non-zero magnitude
                tensor1_flat = sliced_tensor1.flatten()
                tensor2_flat = sliced_tensor2.flatten()
                if tensor1_flat.norm() > 1e-8 and tensor2_flat.norm() > 1e-8:
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()
                else:
                    cosine_sim = 1.0 if torch.allclose(tensor1_flat, tensor2_flat, atol=1e-8) else 0.0
                
                # Replace NaN values with safe defaults
                abs_diff_mean = 0.0 if not torch.isfinite(torch.tensor(abs_diff_mean)) else abs_diff_mean
                abs_diff_max = 0.0 if not torch.isfinite(torch.tensor(abs_diff_max)) else abs_diff_max
                rel_diff_mean = 0.0 if not torch.isfinite(torch.tensor(rel_diff_mean)) else rel_diff_mean
                mse = 0.0 if not torch.isfinite(torch.tensor(mse)) else mse
                cosine_sim = 1.0 if not torch.isfinite(torch.tensor(cosine_sim)) else cosine_sim
                
                # Calculate individual tensor statistics
                tensor1_mean = sliced_tensor1.mean().item()
                tensor1_std = sliced_tensor1.std().item()
                tensor1_min = sliced_tensor1.min().item()
                tensor1_max = sliced_tensor1.max().item()
                
                tensor2_mean = sliced_tensor2.mean().item()
                tensor2_std = sliced_tensor2.std().item()
                tensor2_min = sliced_tensor2.min().item()
                tensor2_max = sliced_tensor2.max().item()
                
                result['summary_stats'] = {
                    'tensor1_stats': {
                        'mean': tensor1_mean,
                        'std': tensor1_std,
                        'min': tensor1_min,
                        'max': tensor1_max
                    },
                    'tensor2_stats': {
                        'mean': tensor2_mean,
                        'std': tensor2_std,
                        'min': tensor2_min,
                        'max': tensor2_max
                    },
                    'difference_stats': {
                        'abs_diff_mean': abs_diff_mean,
                        'abs_diff_max': abs_diff_max,
                        'rel_diff_mean': rel_diff_mean,
                        'mse': mse,
                        'cosine_sim': cosine_sim
                    },
                    'comparison_info': {
                        'elements_compared': min(sliced_tensor1.numel(), sliced_tensor2.numel()),
                        'shapes_match': sliced_tensor1.shape == sliced_tensor2.shape,
                        'values_identical': abs_diff_max < 1e-6
                    }
                }
                
            except Exception as stats_error:
                print(f"Error calculating summary statistics: {stats_error}")
                result['summary_stats'] = {'error': f'Could not calculate statistics: {str(stats_error)}'}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error with manual tensor slicing: {str(e)}'}), 400

def slice_tensor_manually(tensor, indices, count):
    """Apply manual slicing to a tensor with specific dimension indices"""
    # Ensure tensor is on CPU
    tensor = tensor.cpu() if hasattr(tensor, 'cpu') else tensor
    
    if len(indices) == 0:
        # No indices provided, return flattened view
        flat_tensor = tensor.flatten()
        end_idx = min(count, flat_tensor.shape[0])
        return flat_tensor[:end_idx], f'flattened[0:{end_idx}]'
    
    if len(indices) != len(tensor.shape):
        return tensor.flatten()[:count], 'fallback_flattened'
    
    # Build slice objects for each dimension
    slices = []
    slice_parts = []
    
    for i, idx in enumerate(indices):
        dim_size = tensor.shape[i]
        idx = max(0, min(idx, dim_size - 1))  # Clamp to valid range
        
        if i == len(indices) - 1:
            # Last dimension - slice a range of values
            start_idx = idx
            end_idx = min(start_idx + count, dim_size)
            slices.append(slice(start_idx, end_idx))
            slice_parts.append(f'dim{i}[{start_idx}:{end_idx}]')
        else:
            # Other dimensions - use specific index
            slices.append(idx)
            slice_parts.append(f'dim{i}[{idx}]')
    
    # Apply the slicing
    slice_tuple = tuple(slices)
    sliced_tensor = tensor[slice_tuple]
    slice_info = ' × '.join(slice_parts)
    
    return sliced_tensor, slice_info

@app.route('/reorder_matches', methods=['POST'])
def reorder_matches():
    """Update the order of matches based on client-side reordering"""
    data = request.json
    old_index = data.get('old_index')
    new_index = data.get('new_index')
    
    if old_index is None or new_index is None:
        return jsonify({'error': 'old_index and new_index are required'}), 400
    
    global stored_matches
    
    if old_index < 0 or old_index >= len(stored_matches) or new_index < 0 or new_index >= len(stored_matches):
        return jsonify({'error': 'Invalid indices'}), 400
    
    try:
        # Reorder the stored matches
        item = stored_matches.pop(old_index)
        stored_matches.insert(new_index, item)
        
        # Update match indices to reflect new order
        for i, match in enumerate(stored_matches):
            if 'match_index' in match:
                match['match_index'] = i
        
        return jsonify({
            'success': True,
            'message': f'Moved match from position {old_index} to {new_index}',
            'new_order': [match.get('stage', f'Match {i}') for i, match in enumerate(stored_matches)]
        })
        
    except Exception as e:
        return jsonify({'error': f'Error reordering matches: {str(e)}'}), 500

@app.route('/regroup_matches', methods=['POST']) 
def regroup_matches():
    """Regroup matches based on specified strategy"""
    data = request.json
    group_by = data.get('group_by', 'layer')
    
    global stored_matches
    
    try:
        # This endpoint can be used for server-side regrouping if needed
        # For now, we'll just acknowledge the request since regrouping is handled client-side
        return jsonify({
            'success': True,
            'message': f'Regrouped by {group_by}',
            'group_by': group_by
        })
        
    except Exception as e:
        return jsonify({'error': f'Error regrouping matches: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)