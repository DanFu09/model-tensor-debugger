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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

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
            
            # Try multiple loading strategies for compatibility
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

def match_tensors(model1_data, model2_data):
    """Match tensors between two models based on layer and stage order"""
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

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'model1' not in request.files or 'model2' not in request.files:
        return jsonify({'error': 'Both model files required'}), 400
    
    model1_file = request.files['model1']
    model2_file = request.files['model2']
    
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
        
        # Match tensors
        matches = match_tensors(model1_data, model2_data)
        
        # Store matches globally for tensor inspection
        global stored_matches
        stored_matches = matches
        
        # Remove tensors from response for performance
        matches_for_response = []
        for match in matches:
            match_copy = match.copy()
            match_copy.pop('tensor1', None)
            match_copy.pop('tensor2', None)
            matches_for_response.append(match_copy)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'model1_files': len(model1_data),
            'model2_files': len(model2_data),
            'matches': matches_for_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Global storage for tensor data (in production, use proper session management)
stored_matches = []

@app.route('/get_tensor_values', methods=['POST'])
def get_tensor_values():
    data = request.json
    match_index = data.get('match_index', 0)
    start_index = data.get('start_index', 0)
    count = data.get('count', 10)
    
    if match_index >= len(stored_matches):
        return jsonify({'error': 'Invalid match index'}), 400
    
    match = stored_matches[match_index]
    tensor1 = match['tensor1']
    tensor2 = match['tensor2']
    
    try:
        # Determine which tensors are available
        has_tensor1 = tensor1 is not None
        has_tensor2 = tensor2 is not None
        
        # Apply TP-aware reshaping if both tensors available and shapes differ
        if has_tensor1 and has_tensor2 and tensor1.shape != tensor2.shape:
            try:
                tensor1, tensor2 = smart_reshape_for_tp(tensor1, tensor2)
                print(f"  Reshaped tensors for value display: {match['model1_file']} and {match['model2_file']}")
            except Exception as e:
                print(f"  Could not reshape tensors for value display: {e}")
        
        # Use the available tensor to determine shape and processing
        primary_tensor = tensor1 if has_tensor1 else tensor2
        
        # Handle scalars and different tensor shapes for display
        if not hasattr(primary_tensor, 'shape') or len(primary_tensor.shape) == 0:
            # Handle scalar values
            values1 = [float(tensor1)] if has_tensor1 else None
            values2 = [float(tensor2)] if has_tensor2 else None
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'start_index': 0,
                'tensor_shape': [],
                'display_type': 'scalar',
                'argmax_index': None  # No argmax for scalars
            })
        elif len(primary_tensor.shape) == 1:
            # 1D tensor
            end_idx = min(start_index + count, primary_tensor.shape[0])
            
            values1 = tensor1[start_index:end_idx].tolist() if has_tensor1 else None
            values2 = tensor2[start_index:end_idx].tolist() if has_tensor2 else None
            
            # Calculate argmax of absolute differences if both tensors available
            argmax_idx = None
            if has_tensor1 and has_tensor2:
                abs_diff = torch.abs(tensor1 - tensor2)
                argmax_idx = int(torch.argmax(abs_diff).item())
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'start_index': start_index,
                'tensor_shape': list(primary_tensor.shape),
                'display_type': '1d',
                'argmax_index': argmax_idx
            })
        
        elif len(primary_tensor.shape) == 2:
            # 2D tensor - flatten for linear indexing
            flat1 = tensor1.flatten() if has_tensor1 else None
            flat2 = tensor2.flatten() if has_tensor2 else None
            primary_flat = flat1 if has_tensor1 else flat2
            
            end_idx = min(start_index + count, primary_flat.shape[0])
            values1 = flat1[start_index:end_idx].tolist() if has_tensor1 else None
            values2 = flat2[start_index:end_idx].tolist() if has_tensor2 else None
            
            # Calculate argmax of absolute differences if both tensors available
            argmax_idx = None
            if has_tensor1 and has_tensor2:
                abs_diff = torch.abs(flat1 - flat2)
                argmax_idx = int(torch.argmax(abs_diff).item())
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'start_index': start_index,
                'tensor_shape': list(primary_tensor.shape),
                'display_type': '2d (flattened)',
                'argmax_index': argmax_idx
            })
        
        else:
            # Multi-dimensional tensor - flatten for display
            flat1 = tensor1.flatten() if has_tensor1 else None
            flat2 = tensor2.flatten() if has_tensor2 else None
            primary_flat = flat1 if has_tensor1 else flat2
            
            end_idx = min(start_index + count, primary_flat.shape[0])
            values1 = flat1[start_index:end_idx].tolist() if has_tensor1 else None
            values2 = flat2[start_index:end_idx].tolist() if has_tensor2 else None
            
            # Calculate argmax of absolute differences if both tensors available
            argmax_idx = None
            if has_tensor1 and has_tensor2:
                abs_diff = torch.abs(flat1 - flat2)
                argmax_idx = int(torch.argmax(abs_diff).item())
            
            return jsonify({
                'tensor1_values': values1,
                'tensor2_values': values2,
                'start_index': start_index,
                'tensor_shape': list(primary_tensor.shape),
                'display_type': 'multi-dimensional (flattened)',
                'argmax_index': argmax_idx
            })
            
    except Exception as e:
        return jsonify({'error': f'Error extracting tensor values: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)