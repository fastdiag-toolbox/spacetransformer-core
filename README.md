# SpaceTransformer Core

Pure NumPy implementation of SpaceTransformer for geometric computations and 3D medical image transformations.

## Overview

SpaceTransformer Core provides the fundamental `Space` concept - a complete description of 3D image geometry that goes beyond traditional "frame" concepts to include shape information.

## Key Features

### 1. Complete Geometric Description
Unlike traditional "frame" concepts, `Space` fully describes the image sampling grid:
```python
from spacetransformer.core import Space

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

## Core Components

- **Space**: Complete 3D image geometry representation
- **Transform**: 4x4 homogeneous coordinate transformations with lazy inverse computation
- **Point/Vector Operations**: Coordinate transformations and spatial relationship checking

## Format Support

- **DICOM**: `Space.from_dicom(dicom_dataset)`
- **NIfTI**: `Space.from_nifty(nifti_image)`
- **SimpleITK**: `Space.from_sitk(sitk_image)`

## Quick Start

```python
import numpy as np
from spacetransformer.core import Space, Transform

# Create image space
space = Space(
    shape=(512, 512, 100),
    spacing=(1.0, 1.0, 2.0),
    origin=(0, 0, 0)
)

# Define transformations
target_space = space.apply_flip(axis=2).apply_resize((256, 256, 50))

# Get transformation matrix
transform = space.get_transform_to(target_space)
print(f"Transform matrix:\n{transform.matrix}")

# Transform coordinates
points = np.array([[100, 200, 50], [150, 250, 60]])
transformed_points = transform.transform_points(points)
```

## Installation

```bash
pip install spacetransformer-core
```

## Requirements

- Python ≥3.8
- NumPy ≥1.20

## GPU Acceleration

For GPU-accelerated image resampling, install the companion package:
```bash
pip install spacetransformer-torch
```

## License

MIT License

---

*SpaceTransformer Core: Elegant geometric computations for 3D medical images.* 