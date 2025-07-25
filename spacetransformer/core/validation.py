"""Validation utilities for SpaceTransformer parameters.

This module provides validation functions for common SpaceTransformer parameters
including points, point sets, transformation matrices, spaces, and image data.
These functions ensure data integrity and provide clear error messages.

Design Philosophy:
    Provides a set of core validation functions that can be reused across
    the library to ensure consistent error handling. Each function focuses
    on validating a specific data type, with helpful error messages to
    guide users in correcting their inputs.

Example:
    Validating inputs in a function:
    
    >>> from spacetransformer.core.validation import validate_pointset, validate_space
    >>> def warp_points(points, source, target):
    ...     points_np = validate_pointset(points)
    ...     source = validate_space(source)
    ...     target = validate_space(target)
    ...     # Proceed with validated inputs
"""

from typing import Tuple, Union, Optional, Any
import numpy as np

from .exceptions import ValidationError, NumericalError


def validate_point(point: Any, name: str = "point") -> np.ndarray:
    """Validate a single 3D point.
    
    Args:
        point: Input point to validate
        name: Parameter name for error messages
        
    Returns:
        np.ndarray: Validated point as a numpy array
        
    Raises:
        ValidationError: If point is invalid
        
    Example:
        >>> validate_point([1, 2, 3])
        array([1., 2., 3.])
        >>> validate_point([1, 2])  # Will raise ValidationError
    """
    try:
        point_np = np.asarray(point, dtype=float)
    except Exception as e:
        raise ValidationError(f"Could not convert {name} to a numpy array: {e}")
        
    if point_np.size != 3:
        raise ValidationError(
            f"{name} must have exactly 3 elements, got {point_np.size}"
        )
    return point_np.reshape(-1)

def validate_orientation(orientation: Any, name: str = "orientation") -> np.ndarray:
    """Validate a single 3D point.
    
    Args:
        orientation: Input orientation to validate
        name: Parameter name for error messages
        
    Returns:
        np.ndarray: Validated point as a numpy array
        
    Raises:
        ValidationError: If point is invalid
        
    Example:
        >>> validate_orientation([1, 2, 3])
        array([1., 2., 3.])
        >>> validate_orientation([1, 2])  # Will raise ValidationError
    """
    orientation_np = validate_point(orientation, name=name)
    if not np.allclose(np.linalg.norm(orientation_np), 1, atol=1e-6):
        raise ValidationError(f"{name} must be a unit vector")
    return orientation_np


def validate_orthonormal_basis(
    x_orientation: np.ndarray,
    y_orientation: np.ndarray,
    z_orientation: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """Validate that three orientation vectors form an orthonormal basis.
    
    This function checks if the given vectors form a proper orthonormal right-handed basis
    by examining properties of the rotation matrix they form.
    
    Args:
        x_orientation: X-axis orientation vector (unit vector)
        y_orientation: Y-axis orientation vector (unit vector)
        z_orientation: Z-axis orientation vector (unit vector)
        tol: Numerical tolerance for checks
        
    Raises:
        ValidationError: If vectors don't form an orthonormal right-handed basis
    """
    # Form the rotation matrix R
    R = np.column_stack([x_orientation, y_orientation, z_orientation])
    
    # Check orthonormality using R.T @ R = I (identity matrix)
    # This is a single operation that checks all orthogonality conditions
    RTR = R.T @ R
    identity = np.eye(3)
    if not np.allclose(RTR, identity, atol=tol):
        raise ValidationError(
            f"Orientation vectors are not orthonormal. "
            f"R.T @ R should be the identity matrix, but differs by "
            f"{np.max(np.abs(RTR - identity)):.6f}"
        )


def validate_pointset(
    points: Any, name: str = "points", allow_empty: bool = False
) -> np.ndarray:
    """Validate a set of 3D points.
    
    Args:
        points: Input points to validate, should have shape (N, 3)
        name: Parameter name for error messages
        allow_empty: Whether to allow empty point sets
        
    Returns:
        np.ndarray: Validated points as a numpy array with shape (N, 3)
        
    Raises:
        ValidationError: If points are invalid
        
    Example:
        >>> validate_pointset([[1, 2, 3], [4, 5, 6]])
        array([[1., 2., 3.],
               [4., 5., 6.]])
        >>> validate_pointset([1, 2, 3])  # Single point, will be reshaped
        array([[1., 2., 3.]])
    """
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise ValidationError(
            f"{name} must be a list, tuple, or array, got {type(points).__name__}"
        )
    
    try:
        points_np = np.asarray(points, dtype=float)
        
        # Check for empty points first
        if len(points_np) == 0 or points_np.size == 0:
            if not allow_empty:
                raise ValidationError(f"{name} cannot be empty")
            # Return empty array with correct shape (0, 3)
            return np.zeros((0, 3), dtype=float)
            
        # Handle single point case
        if points_np.ndim == 1:
            if points_np.size != 3:
                raise ValidationError(
                    f"Single point {name} must have 3 elements, got {points_np.size}"
                )
            points_np = points_np.reshape(1, 3)
            
        # Validate shape
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValidationError(
                f"{name} must have shape (N,3), got {points_np.shape}"
            )
            
        return points_np
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Could not convert {name} to a valid point set: {e}")


def validate_transform_matrix(
    matrix: Any, name: str = "transform_matrix", check_invertible: bool = True
) -> np.ndarray:
    """Validate a 4x4 transformation matrix.
    
    Args:
        matrix: Input matrix to validate
        name: Parameter name for error messages
        check_invertible: Whether to check if the matrix is invertible
        
    Returns:
        np.ndarray: Validated transformation matrix
        
    Raises:
        ValidationError: If matrix shape is invalid
        NumericalError: If matrix is not invertible (when check_invertible=True)
        
    Example:
        >>> validate_transform_matrix(np.eye(4))
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> validate_transform_matrix(np.eye(3))  # Will raise ValidationError
    """
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.asarray(matrix, dtype=float)
        except Exception as e:
            raise ValidationError(
                f"{name} must be convertible to a numpy array: {e}"
            )
    
    if matrix.shape != (4, 4):
        raise ValidationError(
            f"{name} must be a 4x4 matrix, got {matrix.shape}"
        )
    
    # Check if the bottom row is [0,0,0,1] (valid homogeneous coordinate matrix)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValidationError(
            f"{name} must be a valid homogeneous coordinate matrix "
            f"with bottom row [0,0,0,1], got {matrix[3]}"
        )
    
    # Check if the matrix is invertible
    if check_invertible:
        try:
            det = np.linalg.det(matrix[:3, :3])
            if abs(det) < 1e-10:
                raise NumericalError(
                    f"{name} is nearly singular (determinant={det:.2e}) "
                    f"and may not be invertible"
                )
        except Exception as e:
            if isinstance(e, NumericalError):
                raise
            raise NumericalError(f"Error computing determinant of {name}: {e}")
    
    return matrix


def validate_space(space: Any, name: str = "space") -> "Space":
    """Validate a Space object.
    
    Args:
        space: Input Space object to validate
        name: Parameter name for error messages
        
    Returns:
        Space: The validated Space object
        
    Raises:
        ValidationError: If the input is not a valid Space object
        
    Example:
        >>> from spacetransformer.core import Space
        >>> space = Space(shape=(100, 100, 50))
        >>> validate_space(space) is space
        True
        >>> validate_space("not a space")  # Will raise ValidationError
    """
    from .space import Space  # Import here to avoid circular imports
    
    if not isinstance(space, Space):
        raise ValidationError(
            f"{name} must be a Space object, got {type(space).__name__}"
        )
    
    # The Space class already validates its own parameters on initialization
    return space


def validate_bbox(bbox: Any, check_int: bool = False) -> np.ndarray:
    """Validate a bounding box for space operations.
    
    Args:
        bbox: Input bounding box to validate, should have shape (3, 2)
        check_int: Whether to require integer values in the bbox
        name: Parameter name for error messages
        
    Returns:
        np.ndarray: Validated bbox as a numpy array with shape (3, 2)
        
    Raises:
        ValidationError: If bbox is invalid
        
    Example:
        >>> validate_bbox([[10, 90], [20, 80], [5, 45]])
        array([[10, 90],
               [20, 80],
               [5, 45]])
        >>> validate_bbox([[10.5, 90.2], [20.3, 80.1], [5.0, 45.9]], check_int=False)
        array([[10.5, 90.2],
               [20.3, 80.1],
               [5. , 45.9]])
    """
    # Convert to numpy array
    try:
        bbox_np = np.asarray(bbox)
    except Exception as e:
        raise ValidationError(f"bbox must be convertible to a numpy array: {e}")
        
    # Check shape
    if bbox_np.shape != (3, 2):
        raise ValidationError(f"bbox must be a 3×2 array, got shape {bbox_np.shape}")
    
    # Check integer values if required
    if check_int and not np.all(bbox_np % 1 == 0):
        raise ValidationError(f"bbox must contain only integer values")
    
    # Check bounds
    if not np.all(bbox_np[:, 1] > bbox_np[:, 0]):
        raise ValidationError(f"bbox upper bounds must be greater than lower bounds")
    
    return bbox_np


def validate_image_data(
    image: Any, expected_ndim: Optional[int] = None, name: str = "image"
) -> np.ndarray:
    """Validate image data array.
    
    Args:
        image: Input image data to validate
        expected_ndim: Expected number of dimensions (None to skip check)
        name: Parameter name for error messages
        
    Returns:
        np.ndarray: The validated image data
        
    Raises:
        ValidationError: If the image data is invalid
        
    Example:
        >>> import numpy as np
        >>> image = np.zeros((100, 100, 50))
        >>> validate_image_data(image, expected_ndim=3).shape
        (100, 100, 50)
        >>> validate_image_data(image, expected_ndim=2)  # Will raise ValidationError
    """
    # String and non-array-like types should fail early
    if isinstance(image, (str, bool)):
        raise ValidationError(
            f"{name} must be a numeric array, got {type(image).__name__}"
        )
    
    if not isinstance(image, np.ndarray):
        try:
            image = np.asarray(image)
        except Exception as e:
            raise ValidationError(
                f"{name} must be convertible to a numpy array: {e}"
            )
    
    # Check numeric dtype
    if not np.issubdtype(image.dtype, np.number):
        raise ValidationError(
            f"{name} must have numeric data type, got {image.dtype}"
        )
    
    if expected_ndim is not None and image.ndim != expected_ndim:
        raise ValidationError(
            f"{name} must be a {expected_ndim}-dimensional array, got {image.ndim} dimensions"
        )
    
    if not np.isfinite(image).all():
        raise ValidationError(
            f"{name} contains invalid values (NaN or Infinity)"
        )
    
    return image


def validate_shape(shape: Union[Tuple[int, int, int], list, np.ndarray]) -> None:
    """Validate image shape parameters.
    
    Args:
        shape: Image dimensions as (width, height, depth) or similar sequence
        
    Raises:
        ValidationError: If shape is invalid
        
    Example:
        >>> validate_shape((100, 100, 50))  # No exception raised
        >>> validate_shape([512, 512, 200])  # Lists are also accepted
        >>> validate_shape((100, 100))  # Will raise ValidationError
    """
    if not isinstance(shape, (tuple, list, np.ndarray)):
        raise ValidationError(
            f"Shape must be a tuple, list, or array, got {type(shape).__name__}. "
            f"Expected format: (width, height, depth) with positive integers."
        )
    
    if len(shape) != 3:
        raise ValidationError(
            f"Shape must be a 3-element sequence, got {len(shape)} elements. "
            f"Expected format: (width, height, depth) for 3D medical images."
        )
    
    for i, dim in enumerate(shape):
        if not isinstance(dim, (int, np.integer)):
            raise ValidationError(
                f"Shape dimension {i} must be an integer, got {type(dim).__name__} ({dim}). "
                f"All shape values must be positive integers."
            )
        
        if dim <= 0:
            raise ValidationError(
                f"Shape values must be positive integers, got {shape}. "
                f"Dimension {i} has value {dim}, but all dimensions must be > 0."
            )


def validate_spacing(spacing: Union[Tuple[float, float, float], list, np.ndarray]) -> None:
    """Validate voxel spacing parameters.
    
    Args:
        spacing: Voxel spacing as (x_spacing, y_spacing, z_spacing) in mm
        
    Raises:
        ValidationError: If spacing is invalid
        
    Example:
        >>> validate_spacing((1.0, 1.0, 2.0))  # No exception raised
        >>> validate_spacing([0.5, 0.5, 1.0])  # Lists are also accepted
        >>> validate_spacing((1.0, 1.0))  # Will raise ValidationError
    """
    if not isinstance(spacing, (tuple, list, np.ndarray)):
        raise ValidationError(
            f"Spacing must be a tuple, list, or array, got {type(spacing).__name__}. "
            f"Expected format: (x_spacing, y_spacing, z_spacing) in millimeters."
        )
    
    if len(spacing) != 3:
        raise ValidationError(
            f"Spacing must be a 3-element sequence, got {len(spacing)} elements. "
            f"Expected format: (x_spacing, y_spacing, z_spacing) for 3D medical images."
        )
    
    for i, space in enumerate(spacing):
        if not isinstance(space, (int, float, np.integer, np.floating)):
            raise ValidationError(
                f"Spacing dimension {i} must be a number, got {type(space).__name__} ({space}). "
                f"All spacing values must be positive numbers in millimeters."
            )
        
        if space <= 0:
            raise ValidationError(
                f"Spacing values must be positive numbers, got {spacing}. "
                f"Dimension {i} has value {space}, but all spacing values must be > 0 mm."
            )
        
        if not np.isfinite(space):
            raise ValidationError(
                f"Spacing values must be finite numbers, got {spacing}. "
                f"Dimension {i} has value {space} (inf/nan), which is not valid for physical spacing."
            )