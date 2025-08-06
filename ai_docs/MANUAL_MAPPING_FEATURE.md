# Manual Dimension Mapping Feature

**Added**: August 6, 2025  
**Purpose**: Allow manual control over dimension mapping when comparing tensors with different shapes

## Overview

This feature addresses the issue where query outputs appear to differ even after TP-aware reshaping. By providing manual dimension mapping controls, you can specify exactly which dimensions to slice for each tensor independently, allowing you to verify that tensors truly contain the same values by exploring different dimension combinations.

## How It Works

### Two Modes Available

1. **Auto Mode** (default): Uses the automatic TP-aware reshaping and sliders based on final tensor shape
2. **Manual Mapping Mode**: Lets you control each tensor's dimensions independently

### Manual Mapping Interface

When you click **"Manual Mapping"** in the tensor inspection view:

1. **Original Tensor Shapes**: Shows the original shapes before any TP-aware reshaping
2. **Independent Controls**: Separate dimension controls for each tensor
3. **Flexible Slicing**: You specify the exact indices for each dimension

## Example Use Case

For query tensors that seem different:
```
Tensor 1 (outputs 3): [76, 64, 64]
Tensor 2 (outputs_every_layer 2): [1, 64, 76, 64]
```

**In Manual Mode, you can:**
- Set Tensor 1 dimensions: `dim0=5, dim1=10, dim2=0` (slice along last dim)
- Set Tensor 2 dimensions: `dim0=0, dim1=10, dim2=5, dim3=0` (slice along last dim)
- This lets you map `tensor1[5,10,0:10]` vs `tensor2[0,10,5,0:10]`

## User Interface

### Accessing Manual Mode
1. Upload two model archives
2. Expand a tensor comparison (e.g., query)
3. Click **"Inspect Raw Tensor Values"**
4. Click **"Manual Mapping"** button
5. The interface switches to show original tensor shapes with independent controls

### Controls Available
- **Dimension sliders**: Interactive sliders for each dimension of each tensor (0 to dimension_size-1)
- **Number inputs**: Synchronized number fields that work with the sliders
- **Values to show**: How many values to display from the last dimension
- **Auto-update checkbox**: Automatically applies mapping when sliders change (500ms debounce)
- **Apply Mapping**: Execute the manual slicing (manual trigger)
- **Reset All to 0**: Set all dimension sliders back to 0
- **Copy T1→T2 Settings**: Copy Tensor 1 slider values to Tensor 2 (up to common dimensions)
- **Try Common Mappings**: Helper that suggests common dimension mapping patterns
- **Switch to Auto Mode**: Return to automatic TP-aware slicing

### Results Display
- **Summary Statistics**: Real-time statistics computed for the current mapping
  - Model 1 & 2 means, std deviations, min/max values
  - Max difference and cosine similarity (highlighted)
  - Elements compared and identical values indicator
  - Visual feedback: Green background when values are identical
- **Side-by-side comparison table**
  - Tensor 1 values (blue background)  
  - Tensor 2 values (orange background)
  - Differences (green/red background based on magnitude)
- **Slice information** showing exactly what was extracted
- **Interpretation guidance**: Color-coded messages explaining the results

## Technical Implementation

### New Backend Endpoints

**`/get_tensor_shapes`** - Returns original tensor shapes before reshaping:
```json
{
    "tensor1_shape": [76, 64, 64],
    "tensor1_name": "0_query.pth",
    "tensor2_shape": [1, 64, 76, 64], 
    "tensor2_name": "0_query.pth"
}
```

**`/get_tensor_values_manual`** - Applies manual dimension mapping and computes statistics:
```json
{
    "match_index": 0,
    "tensor1_indices": [5, 10, 0],
    "tensor2_indices": [0, 10, 5, 0],
    "count": 10
}
```

Response includes:
- `tensor1_values`, `tensor2_values`: Sliced tensor values
- `tensor1_slice_info`, `tensor2_slice_info`: Slice descriptions
- `summary_stats`: Complete statistics for the current mapping
  - `tensor1_stats`, `tensor2_stats`: Individual tensor statistics
  - `difference_stats`: Comparison metrics (max diff, cosine similarity, etc.)
  - `comparison_info`: Shape matching, element count, identical values flag

### Key Features
- **Uses original tensors**: Accesses `original_tensor1` and `original_tensor2` stored during upload
- **Independent slicing**: Each tensor uses its own dimension indices
- **Flexible mapping**: You can map any dimension from tensor1 to any dimension from tensor2
- **Safety checks**: Validates indices are within tensor bounds
- **Clear feedback**: Shows exactly which slice was extracted

## Use Cases

### 1. Verifying "Different" Queries Are Actually Same
```
# Auto mode might show different values due to dimension ordering
# Manual mode lets you verify by trying different mappings:
Tensor 1 [76,64,64]: dim0=0, dim1=0, dim2=0 → values starting at [0,0,0:10]
Tensor 2 [1,64,76,64]: dim0=0, dim1=0, dim2=0, dim3=0 → values starting at [0,0,0,0:10]
```

### 2. Exploring Tensor Parallel Layouts
```
# Compare how the same logical data is stored in different TP configurations
TP=1: [sequence_len, hidden_dim] 
TP=8: [sequence_len, tp_rank, hidden_dim/8]
```

### 3. Debugging Dimension Permutations
```
# When tensors have been transposed or reordered:
Original: [batch, seq, heads, head_dim]
Permuted: [seq, batch, heads, head_dim] 
```

## Benefits

1. **Precise Control**: Map any dimension to any other dimension
2. **Verification Tool**: Confirm that "different" tensors actually contain same values
3. **Real-time Statistics**: Get immediate feedback on whether your mapping reveals identical values
4. **Debugging Aid**: Understand how tensor parallel affects data layout
5. **Flexible Exploration**: Try different dimension combinations easily
6. **Clear Feedback**: See exactly what slice is being compared with interpretation guidance

## UI Flow

```
[Inspect Raw Tensor Values] → Auto Mode (default)
    ↓ Click "Manual Mapping"
Manual Mode:
  ┌─ Tensor 1 Controls ─┐  ┌─ Tensor 2 Controls ─┐
  │ Dim 0: 5   ━━━━●──── │  │ Dim 0: 0   ●────────  │
  │        [5] (0-75)   │  │        [0] (0-0)     │
  │ Dim 1: 10  ──●───── │  │ Dim 1: 10  ──●─────  │  
  │        [10] (0-63)  │  │        [10] (0-63)   │
  │ Dim 2: 0   ●─────── │  │ Dim 2: 5   ━━━●───── │
  │        [0] (0-63)   │  │        [5] (0-75)    │
  └─────────────────────┘  │ Dim 3: 0   ●─────────│
                          │        [0] (0-63)    │
  ☑ Auto-update on change  └─────────────────────┘
  [Reset] [Copy T1→T2] [Try Common]
  
    ↓ Sliders move → Auto-update (if enabled) OR Click "Apply Mapping"
    
Summary Statistics (Auto-computed):
  Model 1 Mean: 0.123456    Model 2 Mean: 0.123456
  Max Difference: 0.000000  Cosine Similarity: 1.000000
  Elements Compared: 10     Values Identical: ✓ YES
  🎉 Values are identical! This mapping reveals tensors contain same data.
  
Results Table:
  Index | Tensor 1 | Tensor 2 | Difference
  ─────────────────────────────────────────
    0   | 0.123456 | 0.123456 |  0.000000  ✓
    1   | 0.789012 | 0.789012 |  0.000000  ✓
    ...
```

## Testing

The feature has been tested for:
- ✅ Loading original tensor shapes
- ✅ Manual dimension mapping with various combinations
- ✅ Error handling for invalid indices
- ✅ Scalar tensor handling (no dimensions to map)
- ✅ UI mode switching between auto/manual
- ✅ Results display with difference highlighting

## Next Steps

To use this feature to verify your query tensors:
1. Start the app: `conda activate ml-debug-viz && python app.py`
2. Upload your model archives
3. Navigate to the 0_query comparison
4. Use manual mapping to try different dimension combinations
5. Look for a mapping where the values are identical (differences ~0.000000)

This should help you determine if the query values are truly different or just arranged differently due to tensor parallelism.