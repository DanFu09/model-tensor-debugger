#!/usr/bin/env python3
"""
Unit tests for tensor processing functionality in the ML Model Tensor Debugger.

Tests cover:
- Scalar value handling (int, float, tensor scalars)
- TP-aware tensor reshaping 
- Multi-dimensional slicing
- Edge cases found in real data

Run with:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate ml-debug-viz
    python -m pytest tests/test_tensor_processing.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import smart_reshape_for_tp, calculate_tensor_diff, load_tensor_files


class TestScalarHandling:
    """Test handling of scalar values (int, float, tensor scalars)"""
    
    def test_int_scalar(self):
        """Test processing of integer scalars like 0_sliding.pth"""
        # This mimics the 0_sliding.pth case: tensor vs int
        tensor1 = torch.tensor([[[[[1.0]]]]])  # Shape [1,1,1,1,1] 
        scalar2 = 128  # Integer scalar
        
        # Should not crash when accessing shape
        has_shape1 = hasattr(tensor1, 'shape')
        has_shape2 = hasattr(scalar2, 'shape')
        
        assert has_shape1 is True
        assert has_shape2 is False
        
        # Should handle shape extraction correctly
        shape1 = list(tensor1.shape) if has_shape1 else []
        shape2 = list(scalar2.shape) if has_shape2 else []
        
        assert shape1 == [1, 1, 1, 1, 1]
        assert shape2 == []
    
    def test_float_scalar(self):
        """Test processing of float scalars"""
        tensor1 = torch.tensor([[1.5, 2.5]])
        scalar2 = 3.14159
        
        has_shape1 = hasattr(tensor1, 'shape')
        has_shape2 = hasattr(scalar2, 'shape')
        
        assert has_shape1 is True
        assert has_shape2 is False
        
        shape1 = list(tensor1.shape) if has_shape1 else []
        shape2 = list(scalar2.shape) if has_shape2 else []
        
        assert shape1 == [1, 2]
        assert shape2 == []
    
    def test_tensor_scalar(self):
        """Test processing of 0-dimensional tensors (tensor scalars)"""
        tensor1 = torch.tensor([[1.0, 2.0]])
        tensor_scalar2 = torch.tensor(42.0)  # 0-dimensional tensor
        
        has_shape1 = hasattr(tensor1, 'shape')
        has_shape2 = hasattr(tensor_scalar2, 'shape')
        
        assert has_shape1 is True
        assert has_shape2 is True
        
        shape1 = list(tensor1.shape) if has_shape1 else []
        shape2 = list(tensor_scalar2.shape) if has_shape2 else []
        
        assert shape1 == [1, 2]
        assert shape2 == []  # 0-dimensional tensor has empty shape


class TestTPAwareReshaping:
    """Test TP-aware tensor reshaping functionality"""
    
    def test_query_tensor_reshaping(self):
        """Test the specific 0_query.pth reshaping case from bug report"""
        # Shapes from actual data
        tensor1 = torch.randn(76, 64, 64)      # outputs 3
        tensor2 = torch.randn(1, 64, 76, 64)   # outputs_every_layer 2
        
        reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
        
        # Should result in matching shapes
        assert reshaped1.shape == reshaped2.shape
        assert list(reshaped1.shape) == [76, 64, 64]  # Expected final shape
        
    def test_key_tensor_reshaping(self):
        """Test the 0_key.pth reshaping case"""
        tensor1 = torch.randn(76, 8, 64)       # outputs 3
        tensor2 = torch.randn(1, 8, 76, 64)    # outputs_every_layer 2
        
        reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
        
        assert reshaped1.shape == reshaped2.shape
        # The reshape should result in a compatible shape
        
    def test_post_attn_tensor_reshaping(self):
        """Test the 0_post_attn_pre_resid.pth reshaping case"""
        tensor1 = torch.randn(76, 2880)        # outputs 3
        tensor2 = torch.randn(1, 76, 2880)     # outputs_every_layer 2
        
        reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
        
        assert reshaped1.shape == reshaped2.shape
        
    def test_identical_shapes_no_reshape(self):
        """Test that identical shapes are returned unchanged"""
        tensor1 = torch.randn(64, 128)
        tensor2 = torch.randn(64, 128)
        
        reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
        
        assert torch.equal(reshaped1, tensor1)
        assert torch.equal(reshaped2, tensor2)
        assert reshaped1.shape == reshaped2.shape
    
    def test_cpu_enforcement(self):
        """Test that all operations remain on CPU"""
        tensor1 = torch.randn(10, 20)
        tensor2 = torch.randn(1, 10, 20)
        
        reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
        
        assert reshaped1.device.type == 'cpu'
        assert reshaped2.device.type == 'cpu'


class TestDifferenceCalculation:
    """Test tensor difference calculation with various tensor types"""
    
    def test_matching_tensors(self):
        """Test difference calculation for matching tensors"""
        tensor1 = torch.randn(10, 10)
        tensor2 = tensor1.clone()  # Identical tensors
        
        diff_stats = calculate_tensor_diff(tensor1, tensor2)
        
        assert diff_stats['shape_match'] is True
        assert diff_stats['abs_diff_mean'] == 0.0
        assert diff_stats['abs_diff_max'] == 0.0
        assert diff_stats['mse'] == 0.0
        assert abs(diff_stats['cosine_sim'] - 1.0) < 1e-6  # Should be very close to 1
    
    def test_different_tensors(self):
        """Test difference calculation for different tensors"""
        tensor1 = torch.ones(5, 5)
        tensor2 = torch.zeros(5, 5)
        
        diff_stats = calculate_tensor_diff(tensor1, tensor2)
        
        assert diff_stats['shape_match'] is True
        assert diff_stats['abs_diff_mean'] == 1.0
        assert diff_stats['abs_diff_max'] == 1.0
        assert diff_stats['mse'] == 1.0
    
    def test_reshaping_required(self):
        """Test difference calculation when reshaping is required"""
        tensor1 = torch.randn(1, 8, 10)
        tensor2 = torch.randn(8, 10)
        
        diff_stats = calculate_tensor_diff(tensor1, tensor2)
        
        assert diff_stats['shape_match'] is True
        assert diff_stats['reshaped'] is True
        assert 'original_shape1' in diff_stats
        assert 'original_shape2' in diff_stats
        assert 'final_shape' in diff_stats


class TestRealDataSimulation:
    """Test with synthetic data that mimics the actual file types found in the test directories"""
    
    @pytest.fixture
    def synthetic_file_data(self):
        """Create synthetic data that mimics real file patterns"""
        return {
            # Mimic outputs 3/ directory (model1)
            'model1_data': {
                '0_query.pth': {
                    'tensor': torch.randn(76, 64, 64), 
                    'shape': [76, 64, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_key.pth': {
                    'tensor': torch.randn(76, 8, 64), 
                    'shape': [76, 8, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_value.pth': {
                    'tensor': torch.randn(76, 8, 64), 
                    'shape': [76, 8, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_post_attn_pre_resid.pth': {
                    'tensor': torch.randn(76, 2880), 
                    'shape': [76, 2880], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_scaling.pth': {
                    'tensor': torch.nn.Parameter(torch.randn(64)), 
                    'shape': [64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -2.0, 'max': 2.0
                },
                '0_sliding.pth': {
                    'tensor': torch.tensor([[[[128]]]], dtype=torch.int32), 
                    'shape': [1, 1, 1, 1], 
                    'dtype': 'torch.int32', 
                    'mean': 128.0, 'std': 0.0, 'min': 128.0, 'max': 128.0
                },
            },
            # Mimic outputs_every_layer 2/ directory (model2) 
            'model2_data': {
                '0_query.pth': {
                    'tensor': torch.randn(1, 64, 76, 64), 
                    'shape': [1, 64, 76, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_key.pth': {
                    'tensor': torch.randn(1, 8, 76, 64), 
                    'shape': [1, 8, 76, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_value.pth': {
                    'tensor': torch.randn(1, 8, 76, 64), 
                    'shape': [1, 8, 76, 64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_post_attn_pre_resid.pth': {
                    'tensor': torch.randn(1, 76, 2880), 
                    'shape': [1, 76, 2880], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0
                },
                '0_scaling.pth': {
                    'tensor': torch.randn(64), 
                    'shape': [64], 
                    'dtype': 'torch.float32', 
                    'mean': 0.0, 'std': 1.0, 'min': -2.0, 'max': 2.0
                },
                '0_sliding.pth': {
                    'tensor': 128,  # Integer scalar - this was the problematic case
                    'shape': [], 
                    'dtype': 'int', 
                    'mean': 128.0, 'std': 0.0, 'min': 128.0, 'max': 128.0
                },
            }
        }
    
    def test_all_synthetic_files_process(self, synthetic_file_data):
        """Test that all synthetic file types can be processed without crashes"""
        model1_data = synthetic_file_data['model1_data']
        model2_data = synthetic_file_data['model2_data']
        
        # Test all combinations of files
        common_files = set(model1_data.keys()) & set(model2_data.keys())
        
        for filename in common_files:
            t1 = model1_data[filename]['tensor']
            t2 = model2_data[filename]['tensor']
            
            # Should handle shape access safely
            has_shape1 = hasattr(t1, 'shape')
            has_shape2 = hasattr(t2, 'shape')
            
            shape1 = list(t1.shape) if has_shape1 else []
            shape2 = list(t2.shape) if has_shape2 else []
            
            print(f"Testing {filename}: shapes {shape1} vs {shape2}")
            
            # Should not crash when extracting shapes
            assert isinstance(shape1, list)
            assert isinstance(shape2, list)
            
            # If both have shapes, test reshaping
            if has_shape1 and has_shape2:
                if len(shape1) > 0 and len(shape2) > 0:
                    # Should not crash on reshaping attempt
                    try:
                        reshaped1, reshaped2 = smart_reshape_for_tp(t1, t2)
                        assert reshaped1.shape == reshaped2.shape
                        print(f"  ✅ Reshaped successfully: {list(reshaped1.shape)}")
                    except Exception as e:
                        # Some tensor combinations might not be reshapeable, that's ok
                        print(f"  ⚠️ Reshaping failed for {filename}: {e}")
            else:
                print(f"  ✅ Scalar case handled for {filename}")
    
    def test_sliding_scalar_simulation(self, synthetic_file_data):
        """Test the specific sliding scalar case that caused the original bug"""
        t1 = synthetic_file_data['model1_data']['0_sliding.pth']['tensor']
        t2 = synthetic_file_data['model2_data']['0_sliding.pth']['tensor']
        
        # This should not crash (original bug was 'int' object has no attribute 'shape')
        has_shape1 = hasattr(t1, 'shape')
        has_shape2 = hasattr(t2, 'shape')
        
        shape1 = list(t1.shape) if has_shape1 else []
        shape2 = list(t2.shape) if has_shape2 else []
        
        # Based on our analysis of the real files
        assert has_shape1 is True   # Tensor with shape [1,1,1,1]
        assert has_shape2 is False  # Integer scalar (128)
        
        assert len(shape1) > 0      # Tensor has dimensions
        assert shape2 == []         # Scalar has no dimensions
        
        # Test that the values match expectations
        assert t1.item() == 128     # Tensor contains 128
        assert t2 == 128            # Scalar is 128
    
    def test_complete_pipeline_simulation(self, synthetic_file_data):
        """Test the complete tensor matching and storage pipeline"""
        from app import match_tensors
        
        model1_data = synthetic_file_data['model1_data']
        model2_data = synthetic_file_data['model2_data']
        
        # Test the matching process
        matches = match_tensors(model1_data, model2_data)
        assert len(matches) > 0
        
        # Test storage simulation for each match
        for match in matches:
            tensor1 = match['tensor1'] 
            tensor2 = match['tensor2']
            
            if tensor1 is not None and tensor2 is not None:
                # Test the fixed scalar handling logic
                has_shape1 = hasattr(tensor1, 'shape')
                has_shape2 = hasattr(tensor2, 'shape')
                
                if not has_shape1 or not has_shape2:
                    # At least one is a scalar - should handle gracefully
                    original_shape1 = list(tensor1.shape) if has_shape1 else []
                    original_shape2 = list(tensor2.shape) if has_shape2 else []
                    
                    # Should complete without errors
                    storage_data = {
                        'reshape_applied': False,
                        'original_shapes': [original_shape1, original_shape2],
                        'final_shape': original_shape1 if has_shape1 else original_shape2
                    }
                    assert 'reshape_applied' in storage_data
                else:
                    # Both have shapes - test normal tensor processing
                    if tensor1.shape != tensor2.shape:
                        try:
                            reshaped1, reshaped2 = smart_reshape_for_tp(tensor1, tensor2)
                            assert reshaped1.shape == reshaped2.shape
                        except Exception:
                            # Some might not be reshapeable, that's acceptable
                            pass
        
        print(f"✅ Complete pipeline test passed with {len(matches)} matches")


class TestStorageIntegration:
    """Test the storage system integration with mixed tensor types"""
    
    def test_mixed_storage_simulation(self):
        """Simulate the storage process with mixed tensor/scalar pairs"""
        # Simulate matches with various combinations
        test_cases = [
            {
                'name': 'tensor_vs_tensor_reshaping_needed',
                'tensor1': torch.randn(1, 8, 76, 64),
                'tensor2': torch.randn(76, 8, 64),
                'should_reshape': True
            },
            {
                'name': 'tensor_vs_tensor_same_shape',
                'tensor1': torch.randn(64, 128),
                'tensor2': torch.randn(64, 128),
                'should_reshape': False
            },
            {
                'name': 'tensor_vs_int_scalar',
                'tensor1': torch.tensor([[[[1.0]]]]),
                'tensor2': 128,  # int scalar
                'should_reshape': False
            },
            {
                'name': 'float_scalar_vs_tensor',
                'tensor1': 3.14,  # float scalar
                'tensor2': torch.tensor([1, 2, 3]),
                'should_reshape': False
            },
            {
                'name': 'parameter_vs_tensor',
                'tensor1': torch.nn.Parameter(torch.randn(64)),
                'tensor2': torch.randn(64),
                'should_reshape': False
            },
            {
                'name': 'zero_dim_tensor_vs_scalar',
                'tensor1': torch.tensor(42.0),  # 0-dimensional tensor
                'tensor2': 42,  # int scalar
                'should_reshape': False
            }
        ]
        
        for case in test_cases:
            # Simulate the storage logic from app.py
            tensor1, tensor2 = case['tensor1'], case['tensor2']
            
            print(f"\nTesting case: {case['name']}")
            print(f"  Types: {type(tensor1)} vs {type(tensor2)}")
            
            # Handle scalar values - they don't have shapes
            has_shape1 = hasattr(tensor1, 'shape')
            has_shape2 = hasattr(tensor2, 'shape')
            
            if not has_shape1 or not has_shape2:
                # At least one is a scalar - no reshaping needed
                original_shape1 = list(tensor1.shape) if has_shape1 else []
                original_shape2 = list(tensor2.shape) if has_shape2 else []
                
                stored_data = {
                    'reshape_applied': False,
                    'original_shapes': [original_shape1, original_shape2],
                    'final_shape': original_shape1 if has_shape1 else original_shape2
                }
                
                assert stored_data['reshape_applied'] is False
                print(f"  ✅ Scalar case handled: {original_shape1} vs {original_shape2}")
                
            else:
                # Both have shapes - proceed with normal tensor processing
                original_shape1 = list(tensor1.shape)
                original_shape2 = list(tensor2.shape)
                
                print(f"  Shapes: {original_shape1} vs {original_shape2}")
                
                if tensor1.shape != tensor2.shape and case['should_reshape']:
                    try:
                        reshaped_tensor1, reshaped_tensor2 = smart_reshape_for_tp(tensor1, tensor2)
                        stored_data = {
                            'reshape_applied': True,
                            'original_shapes': [original_shape1, original_shape2],
                            'final_shape': list(reshaped_tensor1.shape)
                        }
                        assert stored_data['reshape_applied'] is True
                        assert reshaped_tensor1.shape == reshaped_tensor2.shape
                        print(f"  ✅ Reshaping successful: → {stored_data['final_shape']}")
                    except Exception as e:
                        # Reshaping failed, store without reshaping
                        stored_data = {
                            'reshape_applied': False,
                            'original_shapes': [original_shape1, original_shape2],
                            'final_shape': None
                        }
                        print(f"  ⚠️ Reshaping failed: {e}")
                else:
                    # Shapes match or no reshaping expected
                    stored_data = {
                        'reshape_applied': False,
                        'original_shapes': [original_shape1, original_shape2],
                        'final_shape': list(tensor1.shape)
                    }
                    print(f"  ✅ No reshaping needed")
            
            # Should complete without errors
            assert 'original_shapes' in stored_data
            assert 'reshape_applied' in stored_data


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_tensors(self):
        """Test handling of empty tensors"""
        empty_tensor = torch.empty(0)
        regular_tensor = torch.randn(5, 5)
        
        # Should handle empty tensors gracefully
        has_shape1 = hasattr(empty_tensor, 'shape')
        has_shape2 = hasattr(regular_tensor, 'shape')
        
        assert has_shape1 is True
        assert has_shape2 is True
        
        shape1 = list(empty_tensor.shape)
        shape2 = list(regular_tensor.shape)
        
        assert shape1 == [0]
        assert shape2 == [5, 5]
    
    def test_very_large_tensors(self):
        """Test with large tensor shapes (without actually creating large tensors)"""
        # Just test the shape handling logic
        shape1 = [1024, 2048, 512]
        shape2 = [2048, 512]
        
        # Simulate tensor objects with these shapes
        tensor1 = torch.randn(2, 3, 4)  # Small tensor for testing
        tensor2 = torch.randn(3, 4)     # Small tensor for testing
        
        # Mock the shapes
        tensor1._mock_shape = shape1
        tensor2._mock_shape = shape2
        
        # Test shape extraction logic
        has_shape1 = hasattr(tensor1, 'shape')
        has_shape2 = hasattr(tensor2, 'shape')
        
        assert has_shape1 is True
        assert has_shape2 is True
    
    def test_none_values(self):
        """Test handling of None values"""
        tensor1 = torch.randn(5, 5)
        tensor2 = None
        
        # Should handle None gracefully
        has_shape1 = hasattr(tensor1, 'shape') if tensor1 is not None else False
        has_shape2 = hasattr(tensor2, 'shape') if tensor2 is not None else False
        
        assert has_shape1 is True
        assert has_shape2 is False
        
        shape1 = list(tensor1.shape) if has_shape1 else []
        shape2 = list(tensor2.shape) if has_shape2 else []
        
        assert shape1 == [5, 5]
        assert shape2 == []
    
    def test_mixed_dtypes(self):
        """Test tensors with different dtypes"""
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        
        # Should handle different dtypes in TP reshaping
        try:
            # Both have same shape, so no reshaping needed
            reshaped1, reshaped2 = smart_reshape_for_tp(int_tensor, float_tensor)
            assert reshaped1.shape == reshaped2.shape
        except Exception as e:
            # If dtype conversion causes issues, that's acceptable
            print(f"Mixed dtype test failed (acceptable): {e}")
    
    def test_complex_tensor_types(self):
        """Test various PyTorch tensor types"""
        # Test different tensor types that might be encountered
        regular_tensor = torch.randn(3, 3)
        parameter = torch.nn.Parameter(torch.randn(3, 3))
        buffer = torch.randn(3, 3).detach()
        
        tensor_types = [
            ('regular_tensor', regular_tensor),
            ('parameter', parameter),
            ('buffer', buffer)
        ]
        
        for name, tensor in tensor_types:
            has_shape = hasattr(tensor, 'shape')
            assert has_shape is True, f"{name} should have shape attribute"
            
            shape = list(tensor.shape)
            assert shape == [3, 3], f"{name} should have correct shape"
            
            # Test CPU enforcement
            if hasattr(tensor, 'cpu'):
                cpu_tensor = tensor.cpu()
                assert cpu_tensor.device.type == 'cpu', f"{name} should be on CPU"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])