"""Point and vector transformation utilities for 3D medical images.

This module provides functions for transforming point sets and vectors between
different medical image coordinate spaces. It supports both NumPy arrays and
PyTorch tensors for GPU acceleration.

Example:
    Transform points between two spaces:
    
    >>> import numpy as np
    >>> from spacetransformer.core import Space
    >>> from spacetransformer.core.pointset_warpers import warp_point
    >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
    >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
    >>> points = np.array([[10, 20, 10], [50, 50, 25]])
    >>> transformed, mask = warp_point(points, source, target)
    >>> print(transformed)
    [[ 5. 10.  5.]
     [25. 25. 12.5]]
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np

try:
    import torch
    _has_torch = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _has_torch = False

from .space import Space
from .transform import Transform


ArrayLike = Union["torch.Tensor", np.ndarray, list, tuple]


def calc_transform(source: Space, target: Space) -> Transform:
    """Calculate transformation matrix from source to target space.
    
    This function computes the transformation that maps voxel coordinates
    from the source space to the target space by chaining the source-to-world
    and world-to-target transformations.
    
    Args:
        source: Source geometric space
        target: Target geometric space
        
    Returns:
        Transform: Transform object representing source.index -> target.index mapping
        
    Example:
        >>> from spacetransformer.core import Space
        >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
        >>> transform = calc_transform(source, target)
        >>> points = np.array([[0, 0, 0], [10, 10, 10]])
        >>> transformed = transform.apply_piont(points)
    """
    mat = target.from_world_transform.matrix @ source.to_world_transform.matrix
    return Transform(mat, source=source, target=target)


def warp_point(
    point_set: ArrayLike,
    source: Space,
    target: Space,
) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
    """Transform point set from source to target space coordinates.
    
    This function transforms a set of points from source voxel coordinates
    to target voxel coordinates and returns a boolean mask indicating which
    points fall within the target space bounds.
    
    Design Philosophy:
        Supports both NumPy and PyTorch tensors with automatic device handling
        to enable seamless integration with both CPU and GPU workflows. The
        output type matches the input type for consistency.
    
    Args:
        point_set: Input points with shape (N, 3) or (3,) for single point
        source: Source geometric space
        target: Target geometric space
        
    Returns:
        Tuple containing:
        - Transformed points in target space coordinates
        - Boolean mask indicating which points are within target bounds
        
    Raises:
        ValueError: If point dimensions are invalid
        
    Example:
        >>> import numpy as np
        >>> from spacetransformer.core import Space
        >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
        >>> points = np.array([[10, 20, 10], [90, 90, 40]])
        >>> transformed, mask = warp_point(points, source, target)
        >>> print(transformed)
        [[ 5. 10.  5.]
         [45. 45. 20.]]
        >>> print(mask)
        [True True]
    """
    istorch = False
    if _has_torch and isinstance(point_set, torch.Tensor):
        device = point_set.device
        istorch = True
        point_set_np = point_set.cpu().numpy()
    else:
        point_set_np = np.asarray(point_set)

    # Handle single point case (3,) -> (1,3)
    if point_set_np.ndim == 1:
        if point_set_np.shape[0] != 3:
            raise ValueError("Single point must be a length-3 array")
        point_set_np = point_set_np[None, :]  # Add batch dimension
    
    assert point_set_np.ndim == 2 and point_set_np.shape[1] == 3, "point_set shape must be (N,3)"

    T = calc_transform(source, target)
    warp_pts = T.apply_piont(point_set_np)

    isin = np.all((warp_pts >= 0) & (warp_pts <= np.array(target.shape)[None] - 1), axis=1)

    if istorch:
        warp_pts_tensor = torch.from_numpy(warp_pts).to(device=device)
        isin_tensor = torch.from_numpy(isin).to(device=device)
        return warp_pts_tensor, isin_tensor
    else:
        return warp_pts, isin


def warp_vector(
    vector_set: ArrayLike,
    source: Space,
    target: Space,
) -> Union["torch.Tensor", np.ndarray]:
    """Transform vector set between coordinate spaces (translation-invariant).
    
    This function transforms vectors (directions) from source to target space
    without applying translation. Only rotational components of the transformation
    are applied since vectors represent directions, not positions.
    
    Args:
        vector_set: Input vectors with shape (N, 3) or (3,) for single vector
        source: Source geometric space
        target: Target geometric space
        
    Returns:
        Transformed vectors in target space coordinates (same type as input)
        
    Raises:
        ValueError: If vector dimensions are invalid
        
    Example:
        >>> import numpy as np
        >>> from spacetransformer.core import Space
        >>> source = Space(shape=(100, 100, 50))
        >>> target = Space(shape=(50, 50, 25))
        >>> vectors = np.array([[1, 0, 0], [0, 1, 0]])
        >>> transformed = warp_vector(vectors, source, target)
        >>> print(transformed)  # Should be unchanged for identity transformation
        [[1. 0. 0.]
         [0. 1. 0.]]
    """
    istorch = False
    if _has_torch and isinstance(vector_set, torch.Tensor):
        device = vector_set.device
        istorch = True
        vec_np = vector_set.cpu().numpy()
    else:
        vec_np = np.asarray(vector_set)

    # Handle single vector case (3,) -> (1,3)
    if vec_np.ndim == 1:
        if vec_np.shape[0] != 3:
            raise ValueError("Single vector must be a length-3 array")
        vec_np = vec_np[None, :]  # Add batch dimension
    
    assert vec_np.ndim == 2 and vec_np.shape[1] == 3

    dtype = vec_np.dtype
    T = calc_transform(source, target)
    warp_vec = T.apply_vector(vec_np)

    if istorch:
        return torch.from_numpy(warp_vec).to(device=device, dtype=vector_set.dtype)
    else:
        return warp_vec.astype(dtype) 