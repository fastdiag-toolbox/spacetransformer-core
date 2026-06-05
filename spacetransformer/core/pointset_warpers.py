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
from typing import Any, Tuple, Union
import numpy as np

from .space import Space
from .transform import Transform


ArrayLike = Union["torch.Tensor", np.ndarray, list, tuple]


def _torch_if_tensor(value: Any):
    """Lazily import torch only when the input is already a torch tensor."""
    value_type = type(value)
    if not value_type.__module__.startswith("torch"):
        return None
    import torch

    return torch if isinstance(value, torch.Tensor) else None


def _validate_torch_pointset(tensor: "torch.Tensor", *, name: str) -> "torch.Tensor":
    from .exceptions import ValidationError

    if tensor.ndim == 1:
        if tensor.shape[0] != 3:
            raise ValidationError(f"{name} single point must have length 3")
        tensor = tensor[None, :]
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValidationError(f"{name} shape must be (N,3)")
    return tensor


def _torch_matrix(matrix: np.ndarray, reference: "torch.Tensor") -> "torch.Tensor":
    import torch

    dtype = reference.dtype if torch.is_floating_point(reference) else torch.get_default_dtype()
    return torch.as_tensor(matrix, dtype=dtype, device=reference.device)


def _apply_torch_transform(points: "torch.Tensor", transform: Transform, *, w: float) -> "torch.Tensor":
    import torch

    matrix = _torch_matrix(transform.matrix, points)
    points = points.to(dtype=matrix.dtype)
    w_col = torch.full((points.shape[0], 1), w, dtype=matrix.dtype, device=points.device)
    points_h = torch.cat([points, w_col], dim=1)
    return (points_h @ matrix.T)[:, :3]


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
        >>> transformed = transform.apply_point(points)
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
        ValidationError: If inputs are invalid
        
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
    from .validation import validate_pointset, validate_space
    
    # Validate inputs
    source = validate_space(source, name="source")
    target = validate_space(target, name="target")
    
    torch_mod = _torch_if_tensor(point_set)
    if torch_mod is not None:
        point_set_tensor = _validate_torch_pointset(point_set, name="point_set")
        T = calc_transform(source, target)
        warp_pts = _apply_torch_transform(point_set_tensor, T, w=1.0)
        target_shape = torch_mod.as_tensor(target.shape, dtype=warp_pts.dtype, device=warp_pts.device)
        isin = torch_mod.all((warp_pts >= 0) & (warp_pts <= target_shape[None] - 1), dim=1)
        return warp_pts, isin

    point_set_np = np.asarray(point_set)

    # Validate point set
    try:
        if point_set_np.ndim == 1:
            if point_set_np.shape[0] != 3:
                raise ValueError("Single point must be a length-3 array")
            point_set_np = point_set_np[None, :]  # Add batch dimension
        
        assert point_set_np.ndim == 2 and point_set_np.shape[1] == 3, "point_set shape must be (N,3)"
    except (ValueError, AssertionError) as e:
        # Convert to standard ValidationError
        from .validation import validate_pointset
        point_set_np = validate_pointset(point_set_np)

    T = calc_transform(source, target)
    warp_pts = T.apply_point(point_set_np)

    isin = np.all((warp_pts >= 0) & (warp_pts <= np.array(target.shape)[None] - 1), axis=1)

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
        ValidationError: If inputs are invalid
        
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
    from .validation import validate_pointset, validate_space
    
    # Validate inputs
    source = validate_space(source, name="source")
    target = validate_space(target, name="target")
    
    torch_mod = _torch_if_tensor(vector_set)
    if torch_mod is not None:
        vec_tensor = _validate_torch_pointset(vector_set, name="vector_set")
        T = calc_transform(source, target)
        return _apply_torch_transform(vec_tensor, T, w=0.0)

    vec_np = np.asarray(vector_set)

    # Validate vector set
    try:
        if vec_np.ndim == 1:
            if vec_np.shape[0] != 3:
                raise ValueError("Single vector must be a length-3 array")
            vec_np = vec_np[None, :]  # Add batch dimension
        
        assert vec_np.ndim == 2 and vec_np.shape[1] == 3
    except (ValueError, AssertionError) as e:
        # Convert to standard ValidationError
        vec_np = validate_pointset(vec_np, name="vector_set")

    dtype = vec_np.dtype
    T = calc_transform(source, target)
    warp_vec = T.apply_vector(vec_np)

    return warp_vec.astype(dtype) 
