# Known Bugs and Issues

This document tracks known bugs, limitations, and planned fixes for the ML Model Tensor Debugger.

**Note:** Fixed bugs have been moved to [ai_docs/FIXED_BUGS.md](ai_docs/FIXED_BUGS.md).

---

## 🐛 Active Bugs

### .pth File Comparisons Are Buggy and Unstable
**Status:** Open  
**Severity:** High  
**Component:** Dual .pth comparison mode  
**Reported:** 2025-01-22

**Description:**
The direct .pth file comparison feature is experimental and frequently exhibits buggy behavior. Users may experience various issues including:
- Unexpected crashes or errors during tensor inspection
- Features not working correctly (tensor value viewing, manual mapping, jump to max diff)
- Inconsistent behavior between sessions
- Interface elements breaking or behaving unexpectedly
- CSS styling issues (buttons getting too large, layout problems)

**Steps to Reproduce:**
1. Upload two .pth files for comparison
2. Try to use advanced features like "Inspect Raw Tensor Values", "Jump to Max Diff", or "Manual Mapping"
3. Observe inconsistent or broken behavior

**Impact:**
- Users cannot reliably use .pth file comparisons for analysis
- May cause confusion or loss of work
- Affects user confidence in the tool
- Users must use archive comparisons as workaround

**Workaround:**
Use archive comparisons instead of direct .pth file comparisons for most reliable results.

**Root Cause:**
The .pth comparison mode was added later and doesn't fully integrate with all existing systems. The codebase has two different rendering paths (archive vs .pth) that aren't properly unified. Recent fixes have improved stability but issues remain.

**To Fix:**
1. Unify the dual .pth template with the archive template completely
2. Ensure all JavaScript functions work correctly with both modes  
3. Add comprehensive error handling for edge cases
4. Test all features thoroughly in .pth mode
5. Fix CSS styling issues and layout problems
6. Remove the experimental warning once stability is achieved

**Priority:** High - this affects a core feature of the application

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