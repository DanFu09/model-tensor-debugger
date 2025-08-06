# How the ML Model Tensor Debugger Works

This document provides a technical overview of the ML Model Tensor Debugger's architecture, data flow, and key components.

## 🏗️ Architecture Overview

The application follows a classic Flask web architecture with a Python backend and vanilla JavaScript frontend, optimized for tensor debugging workflows.

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Frontend (HTML)   │    │   Backend (Flask)    │    │  PyTorch Processing │
│                     │    │                      │    │                     │
│ • Drag & Drop UI    │◄──►│ • File Upload        │◄──►│ • Tensor Loading    │
│ • Multi-Dim Sliders │    │ • Tensor Matching    │    │ • TP-Aware Reshape  │
│ • Color Visualizer  │    │ • Statistical Calc   │    │ • Difference Calc   │
│ • Jump to Max Diff  │    │ • JSON API Responses │    │ • CPU-Only Ops      │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## 📁 File Structure

```
model-tensor-debugger/
├── app.py                 # Main Flask application
├── templates/index.html   # Single-page web interface  
├── screenshots/           # Documentation images
├── ai_docs/              # AI development documentation
├── requirements.txt      # Python dependencies
├── README.md            # User documentation
└── KNOWN_BUGS.md        # Bug tracking and fixes
```

## 🔄 Data Flow

### 1. **File Upload & Processing**
```python
# User uploads two .zip/.tar.gz archives
POST /upload
├── Extract archives to temp directories
├── Scan for .pth tensor files  
├── Load tensors with torch.load(map_location='cpu')
├── Calculate basic statistics (mean, std, min, max)
├── Match tensors by filename patterns (layer_stage.pth)
└── Return metadata + match results
```

### 2. **Tensor Matching Algorithm**
```python
# Pattern: {layer_number}_{stage_name}.pth
# Example: "5_post_attn_pre_resid.pth" → Layer 5, Stage "post_attn_pre_resid"

def match_tensors(model1_data, model2_data):
    # Group by layer number, then match by stage name
    # Handles: both models, model1-only, model2-only cases
    # Sorts by predefined stage order for consistent display
```

### 3. **TP-Aware Tensor Reshaping**
```python
# Handles different tensor parallel configurations
smart_reshape_for_tp(tensor1, tensor2):
├── Check if shapes already match → return as-is
├── Try transpose operations (preserve last dimension)  
├── Apply rank-zero truncation for size mismatches
├── Fallback to flattened view if all else fails
└── Always ensure CPU-only operations
```

### 4. **Multi-Dimensional Slicing**
```javascript
// Frontend: Create sliders based on final tensor shape
POST /get_tensor_values {
  match_index: 0,
  dimension_indices: [5, 12, 0],  // Per-dimension positions
  count: 10,                      // Values to show
  get_argmax: true               // Request max difference location
}

// Backend: Slice tensors using PyTorch indexing
tensor[dim0_idx, dim1_idx, slice(start, end)]
```

## 🧩 Key Components

### Backend Components (`app.py`)

#### **Tensor Loading System**
- **Purpose**: Robust loading of PyTorch tensor files with compatibility handling
- **Features**: Multiple loading strategies, CPU-only mapping, error recovery
- **Challenge**: Handle different PyTorch versions and serialization formats

#### **TP-Aware Reshaping Engine** 
- **Purpose**: Make tensors from different TP configurations comparable
- **Algorithms**: Transpose detection, rank-zero truncation, shape compatibility
- **Innovation**: Preserves semantic meaning while enabling comparison

#### **Multi-Dimensional Slicer**
- **Purpose**: Extract specific tensor slices based on dimension coordinates  
- **Features**: Handles 1D through N-D tensors, validates bounds, CPU-safe operations
- **Challenge**: Map frontend slider positions to PyTorch tensor indexing

#### **Statistical Calculator**
- **Purpose**: Compute difference metrics between tensor pairs
- **Metrics**: MSE, cosine similarity, absolute differences, relative differences
- **Robustness**: NaN/Inf handling, shape validation, CPU-only operations

### Frontend Components (`templates/index.html`)

#### **Dynamic Slider Generator**
- **Purpose**: Create dimension-specific controls based on tensor shapes
- **Features**: Auto-adapts to 1D/2D/3D+ tensors, shows original vs final shapes
- **Innovation**: Maps reshaped tensor dimensions to intuitive UI controls

#### **Color-Coded Visualizer**
- **Purpose**: Visual encoding of tensor values and differences
- **Scheme**: Blue/orange backgrounds (neg/pos), green/orange borders (low/high diff)
- **Algorithm**: Magnitude-based intensity scaling within each batch

#### **Argmax Navigation**
- **Purpose**: Jump directly to maximum difference locations
- **Features**: Multi-dimensional coordinate mapping, automatic slider positioning
- **UX**: Instant navigation with visual feedback and coordinate display

#### **Collapsible Interface Manager**
- **Purpose**: Handle potentially hundreds of tensor comparisons efficiently  
- **Features**: Bulk expand/collapse, floating header, summary statistics
- **Performance**: Lazy loading of tensor values, minimal DOM manipulation

## 🔧 Technical Innovations

### **1. TP-Aware Tensor Compatibility**
The biggest innovation is making tensors from different tensor parallel configurations comparable:

```python
# Example: TP=8 vs TP=1 comparison
# [8, 64, 76, 64] vs [1, 64, 76, 64] → Both become [64, 76, 64]
# Preserves semantic meaning while enabling element-wise comparison
```

### **2. Multi-Dimensional Navigation UI**
Dynamically generates sliders for any tensor dimensionality:

```javascript
// 4D tensor [76, 8, 64, 128] gets:
// Slider 0: Dim 0 (size 76) → specific index
// Slider 1: Dim 1 (size 8)  → specific index  
// Slider 2: Dim 2 (size 64) → specific index
// Slider 3: Dim 3 (size 128) → range of values to display
```

### **3. CPU-Only Tensor Operations**
All tensor operations forced to CPU to avoid CUDA memory issues:

```python
# Multiple safety layers
tensor = torch.load(file_path, map_location='cpu')  # Load to CPU
tensor = tensor.cpu()                               # Explicit CPU move
# All slicing, reshaping, and calculations on CPU
```

### **4. Semantic Tensor Matching**
Intelligent matching based on transformer architecture stages:

```python
stage_order = [
    'post_ln_pre_attn', 'query', 'key', 'value', 
    'post_attn', 'scaling', 'sliding', 'pre_mlp', 'post_mlp'
]
# Matches by semantic meaning, not just filename similarity
```

## 📊 Performance Characteristics

### **Memory Usage**
- **Frontend**: Minimal - only displays current slice values
- **Backend**: Moderate - stores full tensors in memory during session
- **Optimization**: CPU-only operations, incremental processing

### **Speed Bottlenecks**
1. **File Upload**: Archive extraction and tensor loading
2. **Initial Matching**: Computing statistics for all tensor pairs  
3. **TP Reshaping**: Complex reshape operations for large tensors
4. **Multi-Dimensional Slicing**: PyTorch indexing operations

### **Scalability Limits**
- **File Size**: 500MB per archive (configurable)
- **Tensor Count**: Tested with 100+ tensors per model
- **Dimensions**: No theoretical limit, UI tested up to 6D
- **Session Storage**: Limited by available RAM

## 🔄 Request/Response Flow

### **Typical User Session**
```
1. User uploads two model archives
   ├── POST /upload with FormData
   ├── Backend extracts, loads, matches tensors  
   ├── Returns summary + metadata
   └── Frontend displays expandable results

2. User expands a tensor comparison
   ├── Frontend shows pre-computed statistics
   ├── User clicks "Inspect Raw Tensor Values"
   ├── POST /get_tensor_values (get_final_shape=true)
   └── Frontend creates dimension sliders

3. User navigates with sliders
   ├── Each slider change → POST /get_tensor_values
   ├── Backend slices tensors at specified coordinates  
   ├── Returns values + color-coding data
   └── Frontend updates visualization

4. User clicks "Jump to Max Diff"
   ├── POST /get_tensor_values (get_argmax=true)
   ├── Backend computes argmax coordinates
   ├── Returns multi-dimensional coordinates  
   └── Frontend updates all sliders + displays result
```

## 🛠️ Development Patterns

This repository follows **AI-native development patterns**:

- **Incremental feature addition** through conversational development
- **Documentation-driven design** with extensive inline comments
- **Error-first debugging** with comprehensive error handling
- **User feedback integration** through screenshot-driven iteration

See [CONTRIBUTING.md](../CONTRIBUTING.md) for AI-native contribution guidelines.

## 🐛 Known Limitations

1. **Multi-dimensional slicing after complex TP reshaping** - See [KNOWN_BUGS.md](../KNOWN_BUGS.md)
2. **Large tensor memory usage** - All tensors held in server memory
3. **Single-session storage** - No persistence across browser sessions
4. **Limited file format support** - Only .pth files currently supported

## 🚀 Future Architecture Considerations

- **Distributed processing** for very large tensor comparisons
- **Streaming tensor loading** for memory efficiency  
- **WebGL acceleration** for client-side tensor operations
- **Multi-session persistence** with database storage
- **Plugin architecture** for custom tensor matching algorithms