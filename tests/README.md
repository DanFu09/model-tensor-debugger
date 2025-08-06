# Test Suite for ML Model Tensor Debugger

This directory contains comprehensive unit tests for the tensor processing functionality.

## Running Tests

### Environment Setup
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ml-debug-viz
```

### Run All Tests
```bash
python -m pytest tests/test_tensor_processing.py -v
```

### Run Specific Test Classes
```bash
# Test scalar handling only
python -m pytest tests/test_tensor_processing.py::TestScalarHandling -v

# Test TP-aware reshaping only
python -m pytest tests/test_tensor_processing.py::TestTPAwareReshaping -v

# Test edge cases only
python -m pytest tests/test_tensor_processing.py::TestEdgeCases -v
```

## Test Coverage

### TestScalarHandling
- **Purpose**: Test handling of scalar values (int, float, tensor scalars)
- **Key Tests**: 
  - `test_int_scalar`: Integer scalars like `128`
  - `test_float_scalar`: Float scalars like `3.14159`
  - `test_tensor_scalar`: 0-dimensional tensors

### TestTPAwareReshaping  
- **Purpose**: Test TP-aware tensor reshaping functionality
- **Key Tests**:
  - `test_query_tensor_reshaping`: [76,64,64] vs [1,64,76,64] case
  - `test_key_tensor_reshaping`: [76,8,64] vs [1,8,76,64] case
  - `test_cpu_enforcement`: Ensures all operations stay on CPU

### TestDifferenceCalculation
- **Purpose**: Test tensor difference calculation with various tensor types
- **Key Tests**:
  - `test_matching_tensors`: Identical tensors (should have 0 difference)
  - `test_reshaping_required`: Tensors needing reshaping before comparison

### TestRealDataSimulation
- **Purpose**: Test with synthetic data that mimics actual file patterns
- **Key Features**:
  - Uses local synthetic data instead of relying on laptop file paths
  - Simulates all 6 file types found in real data: query, key, value, post_attn, scaling, sliding
  - Tests both tensor/tensor and tensor/scalar combinations
- **Key Tests**:
  - `test_sliding_scalar_simulation`: The specific bug case that caused crashes
  - `test_complete_pipeline_simulation`: Full tensor matching and storage pipeline

### TestStorageIntegration
- **Purpose**: Test storage system with mixed tensor types
- **Key Tests**:
  - 6 different storage scenarios including scalar cases
  - Validates reshape metadata tracking
  - Ensures no crashes with mixed data types

### TestEdgeCases
- **Purpose**: Test edge cases and error conditions
- **Key Tests**:
  - `test_empty_tensors`: Tensors with 0 elements
  - `test_none_values`: Handling None values gracefully
  - `test_mixed_dtypes`: int32 vs float32 tensors
  - `test_complex_tensor_types`: Parameters, buffers, etc.

## Test Results

**Current Status**: ✅ **20/20 tests passing**

- **Scalar handling**: All scalar cases (int, float, tensor scalars) work correctly
- **TP-aware reshaping**: All tensor shape combinations reshape successfully  
- **Storage integration**: Mixed tensor/scalar storage works without crashes
- **Edge cases**: Empty tensors, None values, mixed dtypes handled gracefully

## Synthetic Test Data

The tests use synthetic data that replicates the patterns found in real tensor files:

```python
# Model 1 (mimics "outputs 3/" directory)
'0_query.pth': torch.randn(76, 64, 64)
'0_sliding.pth': torch.tensor([[[[128]]]])  # tensor scalar

# Model 2 (mimics "outputs_every_layer 2/" directory)  
'0_query.pth': torch.randn(1, 64, 76, 64)
'0_sliding.pth': 128  # integer scalar (the bug case!)
```

This eliminates dependency on specific file paths while maintaining test coverage of all real scenarios.

## Integration with CI/CD

These tests can be integrated into continuous integration:

```yaml
# Example GitHub Actions step
- name: Run tensor processing tests
  run: |
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate ml-debug-viz
    python -m pytest tests/test_tensor_processing.py -v --tb=short
```

## Adding New Tests

When adding new tests:

1. **Follow the existing patterns**: Use synthetic data, not file paths
2. **Test both positive and negative cases**: What should work vs what should fail gracefully
3. **Include edge cases**: Empty tensors, None values, unusual dtypes
4. **Document the purpose**: Clear docstrings explaining what each test validates

## Related Documentation

- [BUG_FIX_REPORT.md](../ai_docs/BUG_FIX_REPORT.md): Detailed analysis of the bugs fixed
- [HOW_IT_WORKS.md](../ai_docs/HOW_IT_WORKS.md): Overall system architecture
- [KNOWN_BUGS.md](../KNOWN_BUGS.md): Bug tracking (now shows this issue as fixed)