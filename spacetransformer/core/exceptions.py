"""Exception classes for SpaceTransformer library.

This module provides a simple hierarchy of exception classes for different types
of errors that can occur during medical image geometric transformations and
space operations.

Design Philosophy:
    Uses a lightweight exception hierarchy to provide clear, specific error types
    without adding complexity. Each exception type corresponds to a major category
    of operations in the library, making it easier for users to handle different
    types of errors appropriately.

Example:
    Catching specific exception types:
    
    >>> from spacetransformer.core import Space
    >>> from spacetransformer.core.exceptions import ValidationError
    >>> try:
    ...     space = Space(shape=(-1, 100, 50))  # Invalid negative dimension
    ... except ValidationError as e:
    ...     print(f"Validation failed: {e}")
    Validation failed: Shape values must be positive integers, got (-1, 100, 50)
"""


class SpaceTransformerError(Exception):
    """Base exception for all SpaceTransformer errors.
    
    This is the root exception class for all errors raised by the SpaceTransformer
    library. It provides a common base for catching any SpaceTransformer-related
    error while allowing specific handling of different error types.
    
    All other SpaceTransformer exceptions inherit from this class, enabling
    users to catch all library errors with a single except clause if desired.
    
    Example:
        Catching all SpaceTransformer errors:
        
        >>> from spacetransformer.core.exceptions import SpaceTransformerError
        >>> try:
        ...     # Some SpaceTransformer operation
        ...     pass
        ... except SpaceTransformerError as e:
        ...     print(f"SpaceTransformer operation failed: {e}")
    """
    pass


class ValidationError(SpaceTransformerError):
    """Input validation errors for all SpaceTransformer components.
    
    This exception is raised when any input parameters fail validation checks,
    including Space parameters (shape, spacing, orientation), transformation
    matrices, point sets, and any other user-provided input data.
    
    Common causes include:
    - Invalid dimensions (negative, zero, non-integer values)
    - Incompatible data types
    - Incorrectly shaped arrays
    - Invalid parameter combinations
    
    Example:
        Invalid shape parameters:
        
        >>> from spacetransformer.core import Space
        >>> try:
        ...     space = Space(shape=(0, 100, 50))  # Zero dimension
        ... except ValidationError as e:
        ...     print(f"Validation error: {e}")
        Validation error: Shape values must be positive integers, got (0, 100, 50)
    """
    pass


class NumericalError(SpaceTransformerError):
    """Numerical computation errors during geometric operations.
    
    This exception is raised when numerical issues occur during computations,
    such as singular matrices, numerical instability, or precision problems.
    
    Common causes include:
    - Singular transformation matrices
    - Division by zero
    - Numerical overflow or underflow
    - Insufficient precision causing instability
    
    Example:
        Singular transformation matrix:
        
        >>> import numpy as np
        >>> from spacetransformer.core import Transform
        >>> try:
        ...     # Create a singular matrix
        ...     matrix = np.eye(4)
        ...     matrix[0,0] = 0
        ...     matrix[1,1] = 0
        ...     transform = Transform(matrix)
        ... except NumericalError as e:
        ...     print(f"Numerical error: {e}")
        Numerical error: Transformation matrix is singular and cannot be inverted
    """
    pass


class FormatError(SpaceTransformerError):
    """Medical image format related errors.
    
    This exception is raised when there are issues with medical image format
    handling, including problems with DICOM, NIfTI, or SimpleITK format
    conversions and metadata validation.
    
    Common causes include:
    - Invalid or corrupted medical image files
    - Inconsistent metadata between format specifications
    - Unsupported format features or extensions
    - Missing required metadata fields
    
    Example:
        Invalid NIfTI affine matrix:
        
        >>> # This would be raised when loading a NIfTI file with invalid affine
        >>> # FormatError("NIfTI affine matrix is singular and cannot be inverted. 
        >>> #            Check image orientation and spacing parameters.")
    """
    pass


class CudaError(SpaceTransformerError):
    """CUDA/GPU related errors.
    
    This exception is raised when GPU operations fail, including CUDA runtime
    errors, memory issues, device compatibility problems, or tensor device
    mismatches.
    
    Common causes include:
    - GPU out of memory
    - CUDA device not available
    - Tensor device mismatches
    - GPU kernel failures
    - Driver compatibility issues
    
    Example:
        GPU memory error:
        
        >>> # This would be raised internally when GPU memory is exhausted
        >>> # CudaError("GPU out of memory during image warping. Try reducing 
        >>> #           batch size, using smaller images, or switching to CPU processing.")
    """
    pass