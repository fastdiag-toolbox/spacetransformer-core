"""4x4 homogeneous coordinate transformation matrices for 3D medical images.

This module provides the Transform class for representing and manipulating 4x4
homogeneous coordinate transformation matrices used in 3D medical image processing.
The Transform class handles matrix operations, composition, and coordinate transformations.

Design Philosophy:
    Uses 4x4 homogeneous coordinates to represent both rotation and translation
    in a single matrix operation. This approach simplifies chaining multiple
    transformations and is numerically stable for medical imaging applications.
    
    The class maintains references to source and target spaces for traceability
    and validation, helping prevent coordinate system errors common in medical
    image processing.

Example:
    Basic transformation usage:
    
    >>> import numpy as np
    >>> from spacetransformer.core import Transform
    >>> matrix = np.eye(4)
    >>> transform = Transform(matrix)
    >>> points = [[0, 0, 0], [1, 1, 1]]
    >>> transformed = transform.apply_piont(points)
    >>> print(transformed)
    [[0. 0. 0.]
     [1. 1. 1.]]
     
    Chaining transformations:
    
    >>> T1 = Transform(np.eye(4))
    >>> T2 = Transform(np.eye(4))
    >>> combined = T2 @ T1  # Apply T1 first, then T2
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    # Resolve circular import for type hints
    from .space import Space


ArrayLike = Union[np.ndarray, list, tuple]


def _homogeneous(points: np.ndarray, w: float = 1.0) -> np.ndarray:
    """Convert points to homogeneous coordinates by appending constant w.
    
    This helper function converts 3D points to 4D homogeneous coordinates
    by appending a constant value w to the last dimension.
    
    Args:
        points: Input points array
        w: Homogeneous coordinate value (1.0 for points, 0.0 for vectors)
        
    Returns:
        np.ndarray: Points in homogeneous coordinates with shape (N, 4)
        
    Example:
        >>> import numpy as np
        >>> points = np.array([[1, 2, 3], [4, 5, 6]])
        >>> homogeneous = _homogeneous(points)
        >>> print(homogeneous)
        [[1. 2. 3. 1.]
         [4. 5. 6. 1.]]
    """
    # Convert list/tuple to numpy array
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    
    if points.ndim == 1:
        points = points[None]
    ones = np.full((points.shape[0], 1), w, dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


@dataclass
class Transform:
    """4x4 homogeneous coordinate transformation matrix wrapper.
    
    This class encapsulates a 4x4 transformation matrix and provides methods
    for applying transformations to points and vectors, computing inverses,
    and composing transformations.
    
    Design Philosophy:
        The class is designed purely for geometric coordinate calculations
        without any resampling-related parameters. It maintains references
        to source and target spaces for traceability and validation.
        
        Uses lazy evaluation for expensive operations like matrix inversion
        and caches results for performance. The class is immutable except
        for internal caching to ensure thread safety.
    
    Attributes:
        matrix: 4x4 transformation matrix (source.index → target.index or other)
        source: Source Space object (can be None for world coordinates)
        target: Target Space object (can be None for world coordinates)
        
    Example:
        Creating and using a transformation:
        
        >>> import numpy as np
        >>> from spacetransformer.core import Transform
        >>> matrix = np.array([[1, 0, 0, 10],
        ...                    [0, 1, 0, 20],
        ...                    [0, 0, 1, 30],
        ...                    [0, 0, 0, 1]])
        >>> transform = Transform(matrix)
        >>> point = [0, 0, 0]
        >>> transformed = transform.apply_piont(point)
        >>> print(transformed)
        [[10. 20. 30.]]
    """

    matrix: np.ndarray  # 4×4 matrix (source.index → target.index or other)
    source: Optional["Space"] = None  # Source Space, can be None (e.g., world)
    target: Optional["Space"] = None  # Target Space, can be None
    _inverse_cache: Optional["Transform"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate transformation matrix after initialization.
        
        Raises:
            ValueError: If matrix is not 4x4
        """
        if self.matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4 size")

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def inverse(self) -> "Transform":
        """Return the inverse transformation (lazy computed and cached).
        
        This method computes the inverse transformation matrix and caches
        the result for efficient repeated use. The inverse is computed
        using numpy's linear algebra inverse function.
        
        Returns:
            Transform: Inverse transformation with swapped source/target spaces
            
        Example:
            >>> import numpy as np
            >>> matrix = np.array([[1, 0, 0, 10],
            ...                    [0, 1, 0, 20],
            ...                    [0, 0, 1, 30],
            ...                    [0, 0, 0, 1]])
            >>> transform = Transform(matrix)
            >>> inverse = transform.inverse()
            >>> print(inverse.matrix[0:3, 3])  # Should be [-10, -20, -30]
            [-10. -20. -30.]
        """
        if self._inverse_cache is None:
            inv_mat = np.linalg.inv(self.matrix)
            self._inverse_cache = Transform(inv_mat, source=self.target, target=self.source)
            # Bidirectional caching to avoid redundant inverse computation
            self._inverse_cache._inverse_cache = self
        return self._inverse_cache

    # Matrix multiplication (composition)
    def __matmul__(self, other: "Transform") -> "Transform":
        """Compose two transformations using matrix multiplication.
        
        This method implements the @ operator for transformation composition.
        Following mathematical convention, T2 @ T1 means apply T1 first, then T2.
        
        Args:
            other: Transform to compose with (applied first)
            
        Returns:
            Transform: Composed transformation representing other followed by self
            
        Raises:
            TypeError: If other is not a Transform instance
            
        Example:
            >>> import numpy as np
            >>> T1 = Transform(np.eye(4))  # Identity
            >>> T2 = Transform(np.eye(4))  # Identity
            >>> combined = T2 @ T1
            >>> print(np.allclose(combined.matrix, np.eye(4)))
            True
        """
        if not isinstance(other, Transform):
            raise TypeError("Transform can only be multiplied with Transform (@)")

        new_matrix = self.matrix @ other.matrix  # Apply other first, then self

        return Transform(new_matrix, source=other.source, target=self.target)

    # Equivalent notation: self.compose(other) == other @ self
    def compose(self, other: "Transform") -> "Transform":
        """Compose transformations with self applied first.
        
        This method provides an alternative composition interface where
        self.compose(other) means apply self first, then other.
        
        Args:
            other: Transform to apply after self
            
        Returns:
            Transform: Composed transformation representing self followed by other
            
        Example:
            >>> import numpy as np
            >>> T1 = Transform(np.eye(4))
            >>> T2 = Transform(np.eye(4))
            >>> composed = T1.compose(T2)  # Apply T1 first, then T2
            >>> print(np.allclose(composed.matrix, np.eye(4)))
            True
        """
        return other @ self

    # ------------------------------------------------------------------
    # Apply to points / vectors
    # ------------------------------------------------------------------
    def apply_piont(self, pts: ArrayLike) -> np.ndarray:
        """Apply transformation to a set of points.
        
        This method transforms 3D points using the 4x4 transformation matrix.
        Points are treated as having homogeneous coordinate w=1.0, so they
        are affected by both rotation and translation.
        
        Args:
            pts: Input points with shape (N, 3) or (3,) for single point
            
        Returns:
            np.ndarray: Transformed points with shape (N, 3)
            
        Example:
            >>> import numpy as np
            >>> matrix = np.array([[1, 0, 0, 10],
            ...                    [0, 1, 0, 20],
            ...                    [0, 0, 1, 30],
            ...                    [0, 0, 0, 1]])
            >>> transform = Transform(matrix)
            >>> points = [[0, 0, 0], [1, 1, 1]]
            >>> transformed = transform.apply_piont(points)
            >>> print(transformed)
            [[10. 20. 30.]
             [11. 21. 31.]]
        """
        pts_h = _homogeneous(pts, w=1.0)  # Nx4
        out = (self.matrix @ pts_h.T).T[:, :3]
        return out

    def apply_vector(self, vecs: ArrayLike) -> np.ndarray:
        """Apply transformation to a set of vectors (ignoring translation).
        
        This method transforms 3D vectors using only the rotational part
        of the transformation matrix. Vectors are treated as having
        homogeneous coordinate w=0.0, so they are unaffected by translation.
        
        Args:
            vecs: Input vectors with shape (N, 3) or (3,) for single vector
            
        Returns:
            np.ndarray: Transformed vectors with shape (N, 3)
            
        Example:
            >>> import numpy as np
            >>> matrix = np.array([[1, 0, 0, 10],  # Translation has no effect
            ...                    [0, 1, 0, 20],
            ...                    [0, 0, 1, 30],
            ...                    [0, 0, 0, 1]])
            >>> transform = Transform(matrix)
            >>> vectors = [[1, 0, 0], [0, 1, 0]]
            >>> transformed = transform.apply_vector(vectors)
            >>> print(transformed)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        vecs_h = _homogeneous(vecs, w=0.0)  # Nx4, last component is 0
        out = (self.matrix @ vecs_h.T).T[:, :3]
        return out

    def __call__(self, pts: ArrayLike) -> np.ndarray:
        """Apply transformation to points (callable interface).
        
        This method provides a convenient callable interface for applying
        the transformation to points. Equivalent to apply_piont().
        
        Args:
            pts: Input points with shape (N, 3) or (3,) for single point
            
        Returns:
            np.ndarray: Transformed points with shape (N, 3)
            
        Example:
            >>> import numpy as np
            >>> transform = Transform(np.eye(4))
            >>> points = [[1, 2, 3]]
            >>> transformed = transform(points)  # Same as transform.apply_piont(points)
            >>> print(transformed)
            [[1. 2. 3.]]
        """
        return self.apply_piont(pts)
    
    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return string representation of the Transform.
        
        Returns:
            str: String representation showing matrix and source/target spaces
            
        Example:
            >>> import numpy as np
            >>> transform = Transform(np.eye(4))
            >>> print(repr(transform))
            Transform(matrix=
            [[1. 0. 0. 0.]
             [0. 1. 0. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]],
             source=None, target=None)
        """
        return f"Transform(matrix=\n{self.matrix},\n source={self.source}, target={self.target})" 