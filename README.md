# SpaceTransformer

A Python library for elegant 3D medical image geometric transformations.



## Why SpaceTransformer?

Traditional medical image processing suffers from fragmented coordinate concepts:

- **Frame**: Only captures position/orientation, missing crucial **shape** information
- **NIfTI Affine**: A 4x4 matrix that's hard to interpret - what does each element mean?
- **Manual bookkeeping**: Keeping track of transformations across multiple processing steps

SpaceTransformer introduces the **Space** concept - a complete description of 3D image geometry.

## Key Advantages

### 1. Complete Geometric Description
Unlike traditional "frame" concepts, `Space` fully describes the image sampling grid:
```python
space = Space(
    shape=(512, 512, 100),           # Complete voxel dimensions
    origin=(0.0, 0.0, 0.0),         # Physical position
    spacing=(0.5, 0.5, 2.0),        # Voxel size in mm
    x_orientation=(1, 0, 0),        # Explicit direction vectors
    y_orientation=(0, 1, 0),        # No ambiguous matrix interpretation
    z_orientation=(0, 0, 1)
)
```

### 2. Expressive Spatial Operations
Describe complex transformations explicitly:
```python
# Chain operations elegantly
transformed_space = (space
    .apply_flip(axis=2)
    .apply_resize((256, 256, 50))
    .apply_crop(bbox))
```

### 3. Transparent Matrix Interpretation
No more guessing what a 4x4 affine matrix means - direction vectors are explicit.

## Elegant Image Processing

### 1. Worry-Free Multi-Step Pipelines
```python
# Traditional approach: careful bookkeeping required
# crop -> resize -> segmentation -> resize back -> pad back

# SpaceTransformer approach: fully reversible by design
target_space = original_space.apply_crop(bbox).apply_resize(target_size)
warped_img = warp_image(original_img, 
                        original_space, 
                        target_space, 
                        mode='linear', 
                        pad_value=-1000)
# ... process in target_space ...
segment_result = segmodel(warped_img) # hxwxl, int
keypoint_result = keypmodel(warped_img) # nx3, [i,j,k], index coord
# Automatic optimal path back to original_space
original_segment_result = warp_image(segment_result, 
                                     target_space, 
                                     original_space, 
                                     mode='nearest', 
                                     pad_value = 0)
original_keypoint_result = warp_point(keypoint_result, 
                                      target_space, 
                                      original_space) # nx3, index coord 
world_keypoint_result = target_space.to_world_transform(keypoint_result) # nx3, [x,y,z], mm
```

### 2. Abstract-Then-Execute Pattern
```python
# Plan transformations (fast, no actual image processing)
target_space = source_space.apply_flip(0).apply_rotate(1, 30).apply_crop(bbox).apply_resize((256, 256, 128))

# Execute once with optimal path (GPU-accelerated)
result = warp_image(image, source_space, target_space, cuda_device="cuda:0")
```

### 3. GPU-Accelerated & Deep Learning Ready
- PyTorch backend with automatic optimal transformation paths
- No `align_corners` confusion - transformations are mathematically guaranteed reversible
- Seamless integration with deep learning workflows

## Architecture

- **spacetransformer-core**: Pure NumPy implementation for geometric computations
- **spacetransformer-torch**: GPU-accelerated image resampling with PyTorch

## Quick Start

```python
import numpy as np
from spacetransformer.core import Space
from spacetransformer.torch import warp_image

# Create image space
space = Space(
    shape=(512, 512, 100),
    spacing=(1.0, 1.0, 2.0),
    origin=(0, 0, 0)
)

# Define target transformation
target_space = space.apply_flip(axis=2).apply_resize((256, 256, 50))

# Apply to image (GPU-accelerated)
transformed_image = warp_image(
    image, space, target_space, 
    pad_value=0, cuda_device="cuda:0"
)
```

## Format Support

- **NIfTI**: `Space.from_nifti(nifti_image)`
- **SimpleITK**: `Space.from_sitk(sitk_image)`

## Installation

```bash
pip install spacetransformer-core      # Core functionality
pip install spacetransformer-torch     # GPU acceleration
```

## Requirements

- **Core**: Python ≥3.8, NumPy ≥1.20
- **Torch**: PyTorch ≥1.12 (spacetransformer-torch)

---

*SpaceTransformer: Making 3D medical image transformations as elegant as they should be.* 