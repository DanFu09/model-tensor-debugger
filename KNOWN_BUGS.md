# Known Bugs and Issues

This document tracks known bugs, limitations, and planned fixes for the ML Model Tensor Debugger.

## 🐛 Active Bugs

### Bug #1: Multi-Dimensional Slicing Fails After TP-Aware Reshaping

**Status:** Open  
**Severity:** Medium  
**Affects:** Multi-dimensional tensor navigation with shape mismatches  

**Description:**
When tensors have different shapes that require TP-aware reshaping for compatibility, the multi-dimensional slicing interface fails to work properly. The sliders are created based on the final reshaped tensor dimensions, but the backend slicing operations may still reference the original tensor structures.

**Example Case:**
- Original shapes: `[1, 8, 76, 64]` vs `[76, 8, 64]`  
- Final shape after TP-aware reshaping: `[76, 8, 64]`
- Sliders created for: `[76, 8, 64]` (3 sliders)
- **Problem:** Slicing operations fail despite matching final shapes

**Root Cause Analysis:**
1. The frontend creates sliders based on `result.tensor_shape` from backend
2. The backend applies TP-aware reshaping in `smart_reshape_for_tp()`  
3. However, the original tensor metadata in `globalMatches` may still reference pre-reshape shapes
4. There's a mismatch between the slider dimension indices and the actual tensor slicing logic

**Reproduction Steps:**
1. Load two tensor archives with different but compatible shapes (e.g., `[1, 8, 76, 64]` and `[76, 8, 64]`)
2. Navigate to a tensor pair that requires TP-aware reshaping
3. Click "Inspect Raw Tensor Values"  
4. Try to use the dimension sliders
5. Observe slicing failures or incorrect indexing

**Impact:**
- Users cannot navigate through reshaped multi-dimensional tensors
- Falls back to flattened view with error message
- Reduces debugging effectiveness for TP-incompatible tensor shapes

---

## 🔧 Planned Fixes

### Fix for Bug #1: Multi-Dimensional Slicing After Reshaping

**Implementation Plan:**

#### Phase 1: Backend Consistency (High Priority)
1. **Modify tensor storage**: Store both original AND final reshaped tensors in `stored_matches`
   ```python
   stored_matches[index] = {
       'original_tensor1': original_tensor1,
       'original_tensor2': original_tensor2, 
       'tensor1': reshaped_tensor1,  # Use reshaped versions for slicing
       'tensor2': reshaped_tensor2,
       'reshape_applied': True/False,
       'original_shapes': [shape1, shape2],
       'final_shape': final_shape
   }
   ```

2. **Update `get_tensor_values` endpoint**: Always use the final reshaped tensors for slicing operations
   - Remove redundant reshaping in the endpoint
   - Use pre-reshaped tensors stored in `stored_matches`
   - Ensure dimension indices map to final tensor shapes

3. **Improve reshape logging**: Add detailed logging of reshape operations during upload phase

#### Phase 2: Frontend Robustness (Medium Priority)  
1. **Enhanced error handling**: Better fallback when slicing fails
   - Detect slice operation failures
   - Provide user-friendly error messages
   - Offer alternative views (e.g., different dimension combinations)

2. **Slice validation**: Validate dimension indices against actual tensor shapes before sending to backend

3. **Dynamic slider adjustment**: Update slider ranges based on successful backend responses

#### Phase 3: User Experience (Low Priority)
1. **Visual indicators**: Show when tensors have been reshaped with clear before/after information
2. **Reshape details**: Add a "View Reshape Details" section showing the transformation applied
3. **Alternative slicing modes**: Offer both "original" and "reshaped" tensor navigation options

**Testing Plan:**
1. **Unit tests** for reshape + slicing combinations
2. **Integration tests** with various tensor shape combinations  
3. **User testing** with real tensor parallel model outputs
4. **Regression tests** to ensure existing functionality remains intact

**Estimated Effort:** 4-6 hours
**Priority:** Medium (affects core functionality but has workaround)

---

## 📋 Future Enhancements

### Enhancement Ideas
- **Smart dimension detection**: Auto-detect which dimensions are most interesting to slice
- **Batch slicing**: Allow slicing multiple tensors simultaneously  
- **Dimension correlation**: Show how changes in one dimension affect tensor statistics
- **Export sliced views**: Save specific tensor slices for external analysis

---

## 🧪 Testing Notes

### Test Cases to Add
1. Various TP reshaping scenarios (1D expansion, dimension reordering, etc.)
2. Edge cases with singleton dimensions
3. Large tensor slicing performance
4. Memory usage during reshape operations

### Known Working Configurations  
- ✅ Identical tensor shapes (no reshaping needed)
- ✅ Simple shape differences (e.g., `[64, 64]` vs `[1, 64, 64]`)
- ✅ 1D and 2D tensor navigation
- ✅ Scalar tensor handling

### Problematic Configurations
- ❌ Complex TP reshaping with dimension reordering  
- ❌ Large tensor shape differences requiring multiple transformations
- ❌ Mixed tensor types (some reshaped, some not) in same model comparison