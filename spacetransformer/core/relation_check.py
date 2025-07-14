"""Spatial relationship checking utilities for 3D medical image spaces.

This module provides functions for checking spatial relationships between
different medical image spaces, including overlap detection, alignment checking,
and geometric compatibility validation.

Example:
    Check if two spaces have overlapping regions:
    
    >>> from spacetransformer.core import Space
    >>> from spacetransformer.core.relation_check import find_tight_bbox
    >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
    >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
    >>> bbox = find_tight_bbox(source, target)
    >>> print(bbox.shape)
    (3, 2)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from .space import Space
from .pointset_warpers import warp_point


def _orientation_matrix(space: Space) -> np.ndarray:
    """Extract the 3x3 orientation matrix from a Space object.
    
    Args:
        space: Input Space object
        
    Returns:
        np.ndarray: 3x3 orientation matrix with direction vectors as columns
    """
    return np.column_stack((space.x_orientation, space.y_orientation, space.z_orientation))


def find_tight_bbox(source: Space, target: Space) -> np.ndarray:
    """Calculate the tight bounding box of source space in target index coordinates.
    
    This function computes the minimal bounding box that contains all corners
    of the source space when transformed to the target space's index coordinates.
    The bounding box uses half-open intervals [left, right).
    
    Args:
        source: Source space to compute bounding box for
        target: Target space coordinate system
        
    Returns:
        np.ndarray: Bounding box array of shape (3, 2) where bbox[:,0] contains
                   left bounds and bbox[:,1] contains right bounds (exclusive)
                   
    Example:
        >>> from spacetransformer.core import Space
        >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
        >>> bbox = find_tight_bbox(source, target)
        >>> print(bbox)
        [[ 0 50]
         [ 0 50]
         [ 0 25]]
    """
    s = np.array(source.shape) - 1
    corners = np.array([
        [0, 0, 0],
        [0, 0, s[2]],
        [0, s[1], 0],
        [s[0], 0, 0],
        [s[0], s[1], 0],
        [s[0], 0, s[2]],
        [0, s[1], s[2]],
        [s[0], s[1], s[2]],
    ])

    warp_corners, _ = warp_point(corners, source, target)
    lefts = np.floor(np.min(warp_corners, axis=0))
    rights = np.ceil(np.max(warp_corners, axis=0)) + 1  # Right-open interval

    lefts = np.minimum(np.maximum(lefts, 0), target.shape)
    rights = np.minimum(np.maximum(rights, 0), target.shape)

    tight_bbox = np.stack([lefts, rights]).T.astype(int)
    return tight_bbox


def _check_same_base(source: Space, target: Space) -> bool:
    """Check if two spaces have identical orientation matrices.
    
    Args:
        source: First space to compare
        target: Second space to compare
        
    Returns:
        bool: True if orientation matrices are identical
    """
    return bool(np.all(_orientation_matrix(source) == _orientation_matrix(target)))


def _check_same_spacing(source: Space, target: Space) -> bool:
    """Check if two spaces have identical voxel spacing.
    
    Args:
        source: First space to compare
        target: Second space to compare
        
    Returns:
        bool: True if spacing values are identical
    """
    return bool(np.all(np.array(source.spacing) == np.array(target.spacing)))


def _check_align_corner(source: Space, target: Space) -> bool:
    """Check if target space corners align with source space grid points.
    
    This function verifies whether the target space's origin and end points
    fall exactly on integer grid points when expressed in the source space's
    coordinate system.
    
    Args:
        source: Source space providing the reference grid
        target: Target space to check for alignment
        
    Returns:
        bool: True if target corners align with source grid points
        
    Example:
        >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
        >>> aligned = _check_align_corner(source, target)
        >>> print(aligned)
        True
    """
    # Check if target start/end points fall on source integer grid
    R = _orientation_matrix(source)
    # Column vectors multiplied by corresponding spacing → RS
    M = R * np.array(source.spacing)[None, :]  # 3×3
    offset_origin = np.linalg.solve(M, np.array(target.origin) - np.array(source.origin))
    offset_end = np.linalg.solve(M, np.array(target.end) - np.array(source.origin))
    align_origin = np.linalg.norm(np.round(offset_origin) - offset_origin) < 1e-4
    align_end = np.linalg.norm(np.round(offset_end) - offset_end) < 1e-4
    return bool(align_origin and align_end)


def _check_isin(source: Space, target: Space) -> bool:
    """Check if source space is entirely contained within target space.
    
    This function tests whether all corner points of the source space fall
    within the boundaries of the target space when transformed to world coordinates.
    
    Args:
        source: Source space to check for containment
        target: Target space that might contain the source
        
    Returns:
        bool: True if source is entirely contained within target
        
    Example:
        >>> source = Space(shape=(50, 50, 25), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> contained = _check_isin(source, target)
        >>> print(contained)
        True
    """
    s = np.array(source.shape) - 1
    eight = np.array([
        [0, 0, 0],
        [s[0], 0, 0],
        [0, s[1], 0],
        [0, 0, s[2]],
        [s[0], 0, s[2]],
        [s[0], s[1], 0],
        [0, s[1], s[2]],
        [s[0], s[1], s[2]],
    ])
    eight_world = source.to_world_transform.apply_piont(eight)
    isin = target.contain_pointset_world(eight_world)
    return bool(np.all(isin))


def _check_no_overlap(source: Space, target: Space) -> bool:
    """Check if two spaces have no overlapping regions.
    
    This function determines whether two spaces are spatially disjoint by
    computing their tight bounding boxes in each other's coordinate systems.
    
    Args:
        source: First space to check
        target: Second space to check
        
    Returns:
        bool: True if spaces have no overlap
        
    Example:
        >>> source = Space(shape=(50, 50, 25), origin=(100, 100, 50))
        >>> target = Space(shape=(50, 50, 25), origin=(0, 0, 0))
        >>> no_overlap = _check_no_overlap(source, target)
        >>> print(no_overlap)
        True
    """
    tight = find_tight_bbox(source, target)
    left, right = tight[:, 0], tight[:, 1]
    overlap = np.prod(right - left)
    no1 = bool(overlap == 0)
    tight2 = find_tight_bbox(target, source)
    l2, r2 = tight2[:, 0], tight2[:, 1]
    overlap2 = np.prod(r2 - l2)
    return no1 or bool(overlap2 == 0)


def _check_small_enough(source: Space, target: Space, ratio_thresh: float = 0.2) -> Tuple[bool, np.ndarray]:
    """Check if source space occupies a small fraction of target space.
    
    This function determines whether the source space's bounding box within
    the target space is small relative to the target's total volume.
    
    Args:
        source: Source space to measure
        target: Target space for reference
        ratio_thresh: Threshold ratio below which source is considered small
        
    Returns:
        Tuple containing:
        - bool: True if source is small enough
        - np.ndarray: Tight bounding box of source in target coordinates
        
    Example:
        >>> source = Space(shape=(10, 10, 5))
        >>> target = Space(shape=(100, 100, 50))
        >>> is_small, bbox = _check_small_enough(source, target)
        >>> print(is_small)
        True
    """
    tight = find_tight_bbox(source, target)
    left, right = tight[:, 0], tight[:, 1]
    ratio = np.prod(right - left) / np.prod(target.shape)
    is_small = ratio_thresh > ratio > 0
    return is_small, tight


def _check_valid_flip_permute(source: Space, target: Space) -> Tuple[bool, list, list]:
    """Check if target can be reached from source by flip and permute operations.
    
    This function determines whether the target space can be obtained from the
    source space through a combination of axis flips and permutations. This is
    useful for optimizing image transformations by using simple operations
    instead of general resampling.
    
    Design Philosophy:
        Checks for axis-aligned transformations that can be implemented efficiently
        using tensor operations rather than general interpolation. This optimization
        is crucial for performance in medical image processing pipelines.
    
    Args:
        source: Source space
        target: Target space to reach
        
    Returns:
        Tuple containing:
        - bool: True if target is reachable via flip/permute
        - list: Boolean flags indicating which axes to flip
        - list: Permutation order for axes
        
    Example:
        >>> source = Space(shape=(100, 100, 50))
        >>> target = Space(shape=(50, 100, 100))  # Swap x and z axes
        >>> valid, flips, order = _check_valid_flip_permute(source, target)
        >>> print(valid)
        True
        >>> print(order)
        [2, 1, 0]
    """
    flag = False
    flip_dims = [0, 0, 0]
    transpose_order = [0, 1, 2]
    base_mult = _orientation_matrix(source).T @ _orientation_matrix(target)
    identity_ax = np.abs(np.abs(base_mult) - 1) < 1e-4
    pre, post = np.where(identity_ax)
    sortidx = np.argsort(pre)
    pre, post = pre[sortidx], post[sortidx]
    if len(pre) < 3:
        return flag, flip_dims, transpose_order

    if set(pre.tolist()) == {0, 1, 2} and set(post.tolist()) == {0, 1, 2}:
        if not np.all(post == np.sort(post)):
            flag = True
            transpose_order = post.tolist()
    for i in range(3):
        flip_dims[i] = base_mult[pre[i], post[i]] < 0
    flag = np.any(flip_dims) or flag

    if flag:
        check_space = source.copy()
        for i, fl in enumerate(flip_dims):
            if fl:
                check_space = check_space.apply_flip(i)
        check_space = check_space.apply_permute(transpose_order)
        if check_space != target:
            flag = False
    return flag, flip_dims, transpose_order 