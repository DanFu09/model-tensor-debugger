# Known Bugs and Issues

This document tracks known bugs, limitations, and planned fixes for the ML Model Tensor Debugger.

**Note:** Fixed bugs have been moved to [ai_docs/FIXED_BUGS.md](ai_docs/FIXED_BUGS.md).

---

## 🐛 Active Bugs

*No active bugs currently reported.*

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