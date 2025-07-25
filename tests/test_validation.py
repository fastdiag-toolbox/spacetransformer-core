"""Tests for the validation functions and error handling.

This module tests the validation functions in spacetransformer.core.validation
and ensures that the error handling system works correctly.
"""

import pytest
import numpy as np

from spacetransformer.core.exceptions import ValidationError, NumericalError, SpaceTransformerError
from spacetransformer.core.validation import (
    validate_point,
    validate_pointset,
    validate_transform_matrix,
    validate_space,
    validate_image_data,
    validate_shape,
    validate_spacing
)
from spacetransformer.core import Space


class TestValidationFunctions:
    """Test suite for validation functions."""
    
    def test_validate_point_valid(self):
        """Test validate_point with valid inputs."""
        # Test with different valid formats
        point1 = validate_point([1.0, 2.0, 3.0])
        point2 = validate_point((4.0, 5.0, 6.0))
        point3 = validate_point(np.array([7.0, 8.0, 9.0]))
        
        # Check that they are converted to numpy arrays
        assert isinstance(point1, np.ndarray)
        assert isinstance(point2, np.ndarray)
        assert isinstance(point3, np.ndarray)
        
        # Check values are preserved
        np.testing.assert_array_equal(point1, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(point2, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(point3, [7.0, 8.0, 9.0])
    
    def test_validate_point_invalid(self):
        """Test validate_point with invalid inputs."""
        # Test with wrong type
        with pytest.raises(ValidationError, match="Could not convert point to a numpy array"):
            validate_point("not a point")
        
        # Test with wrong number of elements
        with pytest.raises(ValidationError, match="must have exactly 3 elements"):
            validate_point([1.0, 2.0])
        
        with pytest.raises(ValidationError, match="must have exactly 3 elements"):
            validate_point([1.0, 2.0, 3.0, 4.0])
        
        # Test with non-numeric values
        with pytest.raises(ValidationError, match="Could not convert"):
            validate_point(["a", "b", "c"])
    
    def test_validate_pointset_valid(self):
        """Test validate_pointset with valid inputs."""
        # Test with different valid formats
        points1 = validate_pointset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        points2 = validate_pointset(((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)))
        points3 = validate_pointset(np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]))
        
        # Test with single point (should be reshaped to (1,3))
        points4 = validate_pointset([19.0, 20.0, 21.0])
        
        # Check that they are converted to numpy arrays with correct shape
        assert isinstance(points1, np.ndarray) and points1.shape == (2, 3)
        assert isinstance(points2, np.ndarray) and points2.shape == (2, 3)
        assert isinstance(points3, np.ndarray) and points3.shape == (2, 3)
        assert isinstance(points4, np.ndarray) and points4.shape == (1, 3)
        
        # Check values are preserved
        np.testing.assert_array_equal(points1, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(points2, [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        np.testing.assert_array_equal(points3, [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
        np.testing.assert_array_equal(points4, [[19.0, 20.0, 21.0]])
    
    def test_validate_pointset_invalid(self):
        """Test validate_pointset with invalid inputs."""
        # Test with wrong type
        with pytest.raises(ValidationError, match="must be a list, tuple, or array"):
            validate_pointset("not a pointset")
        
        # Test with wrong shape
        with pytest.raises(ValidationError, match="must have shape"):
            validate_pointset([[1.0, 2.0], [3.0, 4.0]])
        
        # Test with wrong shape (single point)
        with pytest.raises(ValidationError, match="Single point.*must have 3 elements"):
            validate_pointset([1.0, 2.0])
        
        # Test with empty pointset (not allowed by default)
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_pointset([])
        
        # Test empty pointset is allowed when specified
        empty = validate_pointset([], allow_empty=True)
        assert empty.shape[1] == 3  # Should still have 3 columns
        
        # Test with non-numeric values
        with pytest.raises(ValidationError, match="Could not convert.*to a valid point set"):
            validate_pointset([["a", "b", "c"], ["d", "e", "f"]])
    
    def test_validate_transform_matrix_valid(self):
        """Test validate_transform_matrix with valid inputs."""
        # Identity matrix
        matrix1 = validate_transform_matrix(np.eye(4))
        
        # Valid homogeneous transformation
        matrix2 = np.array([
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1]
        ])
        validated2 = validate_transform_matrix(matrix2)
        
        # Valid rotation matrix with translation
        from math import cos, sin
        angle = np.pi/4  # 45 degrees
        c, s = cos(angle), sin(angle)
        matrix3 = np.array([
            [c, -s, 0, 5],
            [s, c, 0, 10],
            [0, 0, 1, 15],
            [0, 0, 0, 1]
        ])
        validated3 = validate_transform_matrix(matrix3)
        
        # Check values are preserved
        np.testing.assert_array_equal(matrix1, np.eye(4))
        np.testing.assert_array_equal(validated2, matrix2)
        np.testing.assert_array_equal(validated3, matrix3)
    
    def test_validate_transform_matrix_invalid(self):
        """Test validate_transform_matrix with invalid inputs."""
        # Test with wrong type
        with pytest.raises(ValidationError, match="must be convertible to a numpy array"):
            validate_transform_matrix("not a matrix")
        
        # Test with wrong shape
        with pytest.raises(ValidationError, match="must be a 4x4 matrix"):
            validate_transform_matrix(np.eye(3))
        
        # Test with invalid homogeneous format (bottom row)
        invalid_matrix = np.eye(4)
        invalid_matrix[3, 3] = 2.0
        with pytest.raises(ValidationError, match="must be a valid homogeneous coordinate matrix"):
            validate_transform_matrix(invalid_matrix)
        
        # Test with singular matrix
        singular_matrix = np.eye(4)
        singular_matrix[0, 0] = 0
        singular_matrix[1, 1] = 0
        with pytest.raises(NumericalError, match="nearly singular"):
            validate_transform_matrix(singular_matrix)
        
        # Test with check_invertible=False (should not raise error)
        validate_transform_matrix(singular_matrix, check_invertible=False)
    
    def test_validate_space_valid(self):
        """Test validate_space with valid Space object."""
        space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        validated = validate_space(space)
        
        # Should return the same object
        assert validated is space
    
    def test_validate_space_invalid(self):
        """Test validate_space with invalid inputs."""
        with pytest.raises(ValidationError, match="must be a Space object"):
            validate_space("not a space")
        
        with pytest.raises(ValidationError, match="must be a Space object"):
            validate_space(None)
        
        with pytest.raises(ValidationError, match="must be a Space object"):
            validate_space(np.eye(4))
    
    def test_validate_image_data_valid(self):
        """Test validate_image_data with valid inputs."""
        # 3D array
        image1 = np.random.rand(10, 20, 30)
        validated1 = validate_image_data(image1, expected_ndim=3)
        
        # 4D array
        image2 = np.random.rand(5, 10, 20, 30)
        validated2 = validate_image_data(image2, expected_ndim=4)
        
        # No dimension check
        image3 = np.random.rand(10, 20)
        validated3 = validate_image_data(image3)  # No expected_ndim
        
        # Check values are preserved
        np.testing.assert_array_equal(validated1, image1)
        np.testing.assert_array_equal(validated2, image2)
        np.testing.assert_array_equal(validated3, image3)
    
    def test_validate_image_data_invalid(self):
        """Test validate_image_data with invalid inputs."""
        # Wrong type
        with pytest.raises(ValidationError, match="must be a numeric array"):
            validate_image_data("not an image")
        
        # Wrong dimensions
        with pytest.raises(ValidationError, match="must be a 3-dimensional array"):
            validate_image_data(np.random.rand(10, 20), expected_ndim=3)
        
        # NaN values
        image_with_nan = np.random.rand(10, 10, 10)
        image_with_nan[5, 5, 5] = np.nan
        with pytest.raises(ValidationError, match="contains invalid values"):
            validate_image_data(image_with_nan)
        
        # Infinity values
        image_with_inf = np.random.rand(10, 10, 10)
        image_with_inf[5, 5, 5] = np.inf
        with pytest.raises(ValidationError, match="contains invalid values"):
            validate_image_data(image_with_inf)
            
        # Non-numeric data
        with pytest.raises(ValidationError, match="must have numeric data type"):
            validate_image_data(np.array(["a", "b", "c"]))
    
    def test_validate_shape_valid(self):
        """Test validate_shape with valid inputs."""
        # Valid shapes
        validate_shape((100, 100, 50))
        validate_shape([200, 300, 400])
        validate_shape(np.array([1, 2, 3]))
    
    def test_validate_shape_invalid(self):
        """Test validate_shape with invalid inputs."""
        # Wrong type
        with pytest.raises(ValidationError, match="must be a tuple, list, or array"):
            validate_shape("not a shape")
        
        # Wrong number of elements
        with pytest.raises(ValidationError, match="must be a 3-element sequence"):
            validate_shape((100, 100))
        
        # Negative values
        with pytest.raises(ValidationError, match="must be positive integers"):
            validate_shape((100, -100, 50))
        
        # Zero values
        with pytest.raises(ValidationError, match="must be positive integers"):
            validate_shape((100, 0, 50))
        
        # Non-integer values
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_shape((100, 100.5, 50))
    
    def test_validate_spacing_valid(self):
        """Test validate_spacing with valid inputs."""
        # Valid spacings
        validate_spacing((1.0, 1.0, 2.0))
        validate_spacing([0.5, 0.5, 1.0])
        validate_spacing(np.array([0.1, 0.2, 0.3]))
    
    def test_validate_spacing_invalid(self):
        """Test validate_spacing with invalid inputs."""
        # Wrong type
        with pytest.raises(ValidationError, match="must be a tuple, list, or array"):
            validate_spacing("not a spacing")
        
        # Wrong number of elements
        with pytest.raises(ValidationError, match="must be a 3-element sequence"):
            validate_spacing((1.0, 1.0))
        
        # Negative values
        with pytest.raises(ValidationError, match="must be positive numbers"):
            validate_spacing((1.0, -1.0, 2.0))
        
        # Zero values
        with pytest.raises(ValidationError, match="must be positive numbers"):
            validate_spacing((1.0, 0.0, 2.0))
        
        # Non-numeric values
        with pytest.raises(ValidationError, match="must be a number"):
            validate_spacing((1.0, "not a number", 2.0))
        
        # Infinite values
        with pytest.raises(ValidationError, match="must be finite numbers"):
            validate_spacing((1.0, np.inf, 2.0))


class TestExceptionHierarchy:
    """Test the exception hierarchy and error propagation."""
    
    def test_exception_inheritance(self):
        """Test the exception class hierarchy."""
        # ValidationError should be a subclass of SpaceTransformerError
        assert issubclass(ValidationError, SpaceTransformerError)
        
        # NumericalError should be a subclass of SpaceTransformerError
        assert issubclass(NumericalError, SpaceTransformerError)
    
    def test_error_propagation(self):
        """Test that errors are properly caught by parent classes."""
        try:
            validate_shape((-1, 100, 50))
            assert False, "Should have raised ValidationError"
        except SpaceTransformerError:
            # Should be caught by the parent class
            pass
        
        try:
            validate_transform_matrix(np.zeros((4, 4)))
            assert False, "Should have raised NumericalError"
        except SpaceTransformerError:
            # Should be caught by the parent class
            pass


class TestIntegration:
    """Integration tests for validation functions with core classes."""
    
    def test_transform_validation(self):
        """Test that Transform class correctly uses validation."""
        from spacetransformer.core import Transform
        
        # Valid matrix should work
        transform = Transform(np.eye(4))
        
        # Invalid matrix should raise ValidationError
        with pytest.raises(ValidationError):
            Transform(np.eye(3))
        
        # Singular matrix should raise NumericalError
        singular = np.eye(4)
        singular[0, 0] = 0
        singular[1, 1] = 0
        with pytest.raises(NumericalError):
            Transform(singular)
    
    def test_warp_point_validation(self):
        """Test that warp_point correctly uses validation."""
        from spacetransformer.core.pointset_warpers import warp_point
        
        source = Space(shape=(100, 100, 50))
        target = Space(shape=(50, 50, 25))
        
        # Valid inputs
        points = np.array([[10, 20, 30], [40, 50, 20]])
        transformed, mask = warp_point(points, source, target)
        
        # Invalid source
        with pytest.raises(ValidationError):
            warp_point(points, "not a space", target)
        
        # Invalid target
        with pytest.raises(ValidationError):
            warp_point(points, source, "not a space")
        
        # Invalid points
        with pytest.raises(ValidationError):
            warp_point([[1, 2], [3, 4]], source, target)  # Wrong shape
    
    def test_warp_vector_validation(self):
        """Test that warp_vector correctly uses validation."""
        from spacetransformer.core.pointset_warpers import warp_vector
        
        source = Space(shape=(100, 100, 50))
        target = Space(shape=(50, 50, 25))
        
        # Valid inputs
        vectors = np.array([[1, 0, 0], [0, 1, 0]])
        transformed = warp_vector(vectors, source, target)
        
        # Invalid source
        with pytest.raises(ValidationError):
            warp_vector(vectors, "not a space", target)
        
        # Invalid target
        with pytest.raises(ValidationError):
            warp_vector(vectors, source, "not a space")
        
        # Invalid vectors
        with pytest.raises(ValidationError):
            warp_vector([[1, 2], [3, 4]], source, target)  # Wrong shape 

    def test_space_initialization_validation(self):
        """Test that Space initialization correctly validates parameters."""
        # Valid initialization
        space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        
        # Invalid shape
        with pytest.raises(ValidationError, match="must be positive integers"):
            Space(shape=(0, 100, 50))
            
        with pytest.raises(ValidationError, match="must be a 3-element sequence"):
            Space(shape=(100, 100))
            
        # Invalid spacing
        with pytest.raises(ValidationError, match="must be positive numbers"):
            Space(shape=(100, 100, 50), spacing=(0.0, 1.0, 2.0))
            
        with pytest.raises(ValidationError, match="must be a 3-element sequence"):
            Space(shape=(100, 100, 50), spacing=(1.0, 2.0)) 