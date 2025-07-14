"""3D medical image geometric space representation and transformations.

This module provides the core Space class for representing medical image geometry
and utilities for coordinate transformations between different spaces. It supports
conversion to/from various medical image formats (DICOM, NIfTI, SimpleITK).

Design Philosophy:
    The Space class uses a right-handed coordinate system with explicit orientation
    vectors to ensure compatibility with medical imaging standards (DICOM, NIfTI).
    This design choice prioritizes correctness over performance for medical applications.
    
    The coordinate system follows the convention where:
    - x-axis points right (patient left to right)
    - y-axis points anterior (patient posterior to anterior)  
    - z-axis points superior (patient inferior to superior)

Example:
    Basic usage of Space class:
    
    >>> from spacetransformer.core import Space
    >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
    >>> print(space.physical_span)
    [99. 99. 98.]
    
    Create space from DICOM-like parameters:
    
    >>> space = Space(
    ...     shape=(512, 512, 100),
    ...     spacing=(0.5, 0.5, 2.0),
    ...     origin=(0, 0, 0)
    ... )
    >>> transform = space.to_world_transform
    >>> points = [[0, 0, 0], [10, 10, 10]]
    >>> world_points = transform.apply_piont(points)
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Union

import numpy as np
from .transform import Transform


@dataclass
class Space:
    """Represents the geometric space of reference for 3D medical images.
    
    This class stores information about the image's position, orientation, spacing,
    and dimensions in physical space. It provides methods for coordinate transformations
    and geometric operations commonly needed in medical image processing.
    
    Design Philosophy:
        Uses explicit orientation vectors instead of implicit axis assumptions to ensure
        compatibility with arbitrary medical image orientations. The design prioritizes
        correctness and traceability over computational efficiency.
        
        The class maintains immutability for safety - all transformation methods return
        new Space instances rather than modifying existing ones. This prevents
        accidental corruption of geometric metadata in medical applications.
    
    Attributes:
        shape: Image dimensions (height, width, depth) in voxels
        origin: Physical coordinates (x,y,z) of the first voxel in mm
        spacing: Physical size (x,y,z) of each voxel in mm
        x_orientation: Direction cosines of axis 0 (x-axis)
        y_orientation: Direction cosines of axis 1 (y-axis)
        z_orientation: Direction cosines of axis 2 (z-axis)
        
    Example:
        Creating a space for a typical CT scan:
        
        >>> space = Space(
        ...     shape=(512, 512, 100),
        ...     spacing=(0.5, 0.5, 2.0),
        ...     origin=(0, 0, 0)
        ... )
        >>> print(space.physical_span)
        [255.5 255.5 198. ]
        
        Transform between index and world coordinates:
        
        >>> index_points = [[0, 0, 0], [10, 10, 10]]
        >>> world_points = space.to_world_transform.apply_piont(index_points)
        >>> back_to_index = space.from_world_transform.apply_piont(world_points)
    """

    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float] = field(default_factory=lambda: (0, 0, 0))
    spacing: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 1))
    x_orientation: Tuple[float, float, float] = field(default_factory=lambda: (1, 0, 0))
    y_orientation: Tuple[float, float, float] = field(default_factory=lambda: (0, 1, 0))
    z_orientation: Tuple[float, float, float] = field(default_factory=lambda: (0, 0, 1))

    def __post_init__(self):
        """Perform type checking and conversion after initialization.
        
        Converts numpy arrays to tuples for JSON serialization and initializes
        cached transformation matrices for efficient repeated use.
        """
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, np.ndarray):
                # Convert numpy arrays to tuples
                object.__setattr__(self, field_name, tuple(val.tolist()))
        # Initialize cached transforms
        object.__setattr__(self, "_to_world_transform", None)
        object.__setattr__(self, "_from_world_transform", None)

    def to_json(self) -> str:
        """Serialize the Space object to a JSON string.
        
        All attributes are already in JSON-serializable types (tuple/list/float/int).

        Returns:
            str: JSON string representation of the Space
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> json_str = space.to_json()
            >>> print('"shape": [100, 100, 50]' in json_str)
            True
        """
        return json.dumps(asdict(self))

    def reverse_axis_order(self) -> "Space":
        """Convert space information to ZYX order for Python indexing.
        
        This method reverses the axis order from XYZ (medical standard) to ZYX
        (Python/NumPy indexing standard). Useful when interfacing with libraries
        that expect ZYX ordering.
            
        Returns:
            Space: New Space instance with axes in ZYX order
            
        Example:
            >>> space = Space(shape=(100, 200, 50), spacing=(1.0, 2.0, 3.0))
            >>> zyx_space = space.reverse_axis_order()
            >>> print(zyx_space.shape)
            (50, 200, 100)
            >>> print(zyx_space.spacing)
            (3.0, 2.0, 1.0)
        """
        new_shape = self.shape[::-1]
        new_origin = self.origin
        new_spacing = self.spacing[::-1]
        new_x_orientation = self.z_orientation
        new_y_orientation = self.y_orientation
        new_z_orientation = self.x_orientation
        return Space(
            new_shape,
            new_origin,
            new_spacing,
            new_x_orientation,
            new_y_orientation,
            new_z_orientation,
        )

    @property
    def shape_zyx(self) -> Tuple[int, int, int]:
        """Get the shape in ZYX order for Python indexing.
        
        Returns:
            Tuple[int, int, int]: Shape in (depth, height, width) order
            
        Example:
            >>> space = Space(shape=(100, 200, 50))  # width, height, depth
            >>> print(space.shape_zyx)  # depth, height, width
            (50, 200, 100)
        """
        return self.shape[::-1]

    @classmethod
    def from_dict(cls, data: dict) -> "Space":
        """Create a Space object from a dictionary.
        
        Lists in the dictionary will be automatically converted to tuples
        to match the expected Space attribute types.
        
        Args:
            data: Dictionary containing Space attributes
                 Lists will be converted to tuples

        Returns:
            Space: A new Space instance
            
        Example:
            >>> data = {
            ...     'shape': [100, 100, 50],
            ...     'spacing': [1.0, 1.0, 2.0],
            ...     'origin': [0, 0, 0]
            ... }
            >>> space = Space.from_dict(data)
            >>> print(space.shape)
            (100, 100, 50)
        """
        # Convert lists to tuples where needed
        converted_data = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in data.items()
        }
        return cls(**converted_data)

    @classmethod
    def from_json(cls, json_str: str) -> "Space":
        """Create a Space object from a JSON string.
        
        Args:
            json_str: JSON string containing Space data

        Returns:
            Space: A new Space instance
            
        Raises:
            json.JSONDecodeError: If the JSON string is invalid
            
        Example:
            >>> json_str = '{"shape": [100, 100, 50], "spacing": [1.0, 1.0, 2.0]}'
            >>> space = Space.from_json(json_str)
            >>> print(space.shape)
            (100, 100, 50)
        """
        obj_dict = json.loads(json_str)
        return cls.from_dict(obj_dict)

    @classmethod
    def from_sitk(cls, simpleitkimage: "SimpleITK.Image") -> "Space":
        """Create a Space object from a SimpleITK Image.
        
        Args:
            simpleitkimage: SimpleITK Image object

        Returns:
            Space: A new Space instance with geometry matching the SimpleITK image
            
        Example:
            >>> import SimpleITK as sitk
            >>> image = sitk.Image(100, 100, 50, sitk.sitkFloat32)
            >>> space = Space.from_sitk(image)
            >>> print(space.shape)
            (100, 100, 50)
        """
        return get_space_from_sitk(simpleitkimage)
    
    @classmethod
    def from_nifti(cls, niftiimage) -> "Space":
        """Create a Space object from a NIfTI image.
        
        Args:
            niftiimage: NIfTI image object

        Returns:
            Space: A new Space instance with geometry matching the NIfTI image
            
        Example:
            >>> import nibabel as nib
            >>> image = nib.load('image.nii.gz')
            >>> space = Space.from_nifti(image)
            >>> print(space.shape)
            (100, 100, 50)
        """
        return get_space_from_nifti(niftiimage)

    def to_sitk_direction(self) -> Tuple[float, ...]:
        """Convert orientation vectors to SimpleITK direction matrix format.
        
        SimpleITK uses a flattened column-major direction matrix representation
        where the direction cosines are stored as a 9-element tuple.

        Returns:
            tuple: Direction cosines in column-major order
                  (xx,yx,zx,xy,yy,zy,xz,yz,zz)
                  
        Example:
            >>> space = Space(shape=(100, 100, 50))
            >>> direction = space.to_sitk_direction()
            >>> print(len(direction))
            9
            >>> print(direction[:3])  # First column (x-orientation)
            (1.0, 0.0, 0.0)
        """
        x = self.x_orientation
        y = self.y_orientation
        z = self.z_orientation
        return (x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2])

    def to_nifti_affine(self) -> np.ndarray:
        """Convert space information to NIfTI affine transformation matrix.
        
        The affine matrix combines rotation, scaling, and translation into a single
        4x4 homogeneous transformation matrix following NIfTI conventions.
        
        Returns:
            np.ndarray: 4x4 affine transformation matrix
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> affine = space.to_nifti_affine()
            >>> print(affine.shape)
            (4, 4)
            >>> print(affine[0, 0])  # x-spacing
            1.0
            >>> print(affine[2, 2])  # z-spacing
            2.0
        """
        # Rotation matrix R from orientation cosines
        R = np.array(
            [self.x_orientation, self.y_orientation, self.z_orientation]
        ).T  # Shape (3, 3)

        # Scaling matrix S from spacing
        S = np.diag(self.spacing)

        # Compute affine transformation matrix
        affine = np.eye(4)
        affine[:3, :3] = np.dot(R, S)
        affine[:3, 3] = self.origin

        return affine

    def to_dicom_orientation(self) -> Tuple[float, ...]:
        """Convert orientation vectors to DICOM Image Orientation (Patient) format.
        
        DICOM stores orientation as a 6-element array containing the direction
        cosines of the first row and first column of the image matrix.

        Returns:
            tuple: Row and column direction cosines concatenated
                  (Xx,Xy,Xz,Yx,Yy,Yz)
                  
        Example:
            >>> space = Space(shape=(100, 100, 50))
            >>> orientation = space.to_dicom_orientation()
            >>> print(len(orientation))
            6
            >>> print(orientation[:3])  # x-orientation
            (1.0, 0.0, 0.0)
            >>> print(orientation[3:])  # y-orientation
            (0.0, 1.0, 0.0)
        """
        return self.x_orientation + self.y_orientation

    def _orientation_matrix(self) -> np.ndarray:
        """Return the 3x3 orientation matrix with direction cosines as columns.
        
        This internal method constructs the orientation matrix used throughout
        the class for coordinate transformations.
        
        Returns:
            np.ndarray: 3x3 matrix with orientation vectors as columns
            
        Example:
            >>> space = Space(shape=(100, 100, 50))
            >>> R = space._orientation_matrix()
            >>> print(R.shape)
            (3, 3)
            >>> print(np.allclose(R, np.eye(3)))  # Identity for default orientation
            True
        """
        return np.column_stack((self.x_orientation, self.y_orientation, self.z_orientation))

    # ------------------------------------------------------------------
    # World ↔ Index transformations
    # ------------------------------------------------------------------
    @property
    def to_world_transform(self) -> Transform:
        """Get the index → world coordinate transformation (lazy-loaded).
        
        This transformation converts voxel indices to world coordinates in mm.
        The transformation is cached for performance since it's frequently used.
        
        Returns:
            Transform: Transform object for index → world coordinate mapping
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> transform = space.to_world_transform
            >>> index_points = [[0, 0, 0], [10, 10, 10]]
            >>> world_points = transform.apply_piont(index_points)
            >>> print(world_points[1])  # Point at index (10, 10, 10)
            [10. 10. 20.]
        """
        if getattr(self, "_to_world_transform") is None:
            RS = self._orientation_matrix() @ np.diag(self.spacing)
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = RS
            mat[:3, 3] = self.origin
            tw = Transform(mat, source=self, target=None)
            object.__setattr__(self, "_to_world_transform", tw)
        return self._to_world_transform  # type: ignore

    @property
    def from_world_transform(self) -> Transform:
        """Get the world → index coordinate transformation (lazy-loaded).
        
        This transformation converts world coordinates in mm to voxel indices.
        The transformation is cached for performance.
        
        Returns:
            Transform: Transform object for world → index coordinate mapping
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> transform = space.from_world_transform
            >>> world_points = [[10.0, 10.0, 20.0]]
            >>> index_points = transform.apply_piont(world_points)
            >>> print(index_points[0])  # Should be [10, 10, 10]
            [10. 10. 10.]
        """
        if getattr(self, "_from_world_transform") is None:
            fw = self.to_world_transform.inverse()
            object.__setattr__(self, "_from_world_transform", fw)
        return self._from_world_transform  # type: ignore

    # ------------------------------------------------------------------
    # Common geometric quantities
    # ------------------------------------------------------------------
    @property
    def physical_span(self) -> np.ndarray:
        """Get the total physical span of the image in world coordinates (mm).
        
        Calculates the physical dimensions of the image volume by applying
        the orientation matrix to the extent vector scaled by spacing.
        
        Returns:
            np.ndarray: Physical span vector in world coordinates (mm)
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> span = space.physical_span
            >>> print(span)
            [99. 99. 98.]
            
            >>> # With rotation, the span changes
            >>> rotated = space.apply_rotate(2, 45, unit='degree')
            >>> rotated_span = rotated.physical_span
            >>> print(rotated_span.shape)
            (3,)
        """
        # (shape-1) per-axis distance vector
        extent = (np.array(self.shape) - 1)
        return self._orientation_matrix() @ (extent * np.array(self.spacing))

    @property
    def end(self) -> np.ndarray:
        """Get the world coordinates of the image's corner voxel.
        
        Calculates the world coordinates of the voxel at the maximum index
        position (shape-1) by adding the physical span to the origin.
        
        Returns:
            np.ndarray: World coordinates of the corner voxel (mm)
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> end_point = space.end
            >>> print(end_point)
            [99. 99. 98.]
            
            >>> # With custom origin
            >>> space_with_origin = Space(shape=(100, 100, 50), 
            ...                          spacing=(1.0, 1.0, 2.0),
            ...                          origin=(10, 20, 30))
            >>> end_point = space_with_origin.end
            >>> print(end_point)
            [109. 119. 128.]
        """
        return np.array(self.origin) + self.physical_span

    # ------------------------------------------------------------------
    # apply_* chain geometric operations
    # ------------------------------------------------------------------
    def apply_flip(self, axis: int) -> "Space":
        """Flip the space along a specified axis.
        
        This method flips the image space along one of the three axes, updating
        both the origin and orientation vectors accordingly.
        
        Args:
            axis: Axis to flip along (0=x, 1=y, 2=z)
            
        Returns:
            Space: New Space instance with flipped axis
            
        Raises:
            AssertionError: If axis is not in {0, 1, 2}
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> flipped = space.apply_flip(0)  # Flip along x-axis
            >>> print(flipped.origin)
            (99.0, 0, 0)
            >>> print(flipped.x_orientation)
            (-1.0, 0.0, 0.0)
        """
        assert axis in (0, 1, 2), "axis should be 0/1/2"
        # Calculate new origin
        new_origin = list(self.origin)
        new_origin[axis] = self.end[axis]
        # Flip orientation vector
        R = self._orientation_matrix().copy()
        R[:, axis] *= -1
        # Split orientation vectors
        x_o, y_o, z_o = (R[:, 0], R[:, 1], R[:, 2])
        return Space(
            shape=self.shape,
            origin=tuple(new_origin),
            spacing=self.spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    def apply_swap(self, axis1: int, axis2: int) -> "Space":
        """Swap two axes in the space.
        
        This method exchanges two axes by reordering the shape, spacing, and
        orientation vectors. Equivalent to apply_permute with a swap permutation.
        
        Args:
            axis1: First axis to swap (0=x, 1=y, 2=z)
            axis2: Second axis to swap (0=x, 1=y, 2=z)
            
        Returns:
            Space: New Space instance with swapped axes
            
        Raises:
            AssertionError: If axes are not in {0, 1, 2} or are equal
            
        Example:
            >>> space = Space(shape=(100, 200, 50), spacing=(1.0, 2.0, 3.0))
            >>> swapped = space.apply_swap(0, 2)  # Swap x and z axes
            >>> print(swapped.shape)
            (50, 200, 100)
            >>> print(swapped.spacing)
            (3.0, 2.0, 1.0)
        """
        assert axis1 in (0, 1, 2) and axis2 in (0, 1, 2) and axis1 != axis2
        order = [0, 1, 2]
        order[axis1], order[axis2] = order[axis2], order[axis1]
        return self.apply_permute(order)

    def apply_permute(self, axis_order: List[int]) -> "Space":
        """Rearrange axes according to the given order.
        
        This method reorders the axes of the space according to the specified
        permutation, updating shape, spacing, and orientation vectors.
        
        Args:
            axis_order: Permutation of [0, 1, 2] specifying the new axis order
            
        Returns:
            Space: New Space instance with reordered axes
            
        Raises:
            AssertionError: If axis_order is not a valid permutation of [0, 1, 2]
            
        Example:
            >>> space = Space(shape=(100, 200, 50), spacing=(1.0, 2.0, 3.0))
            >>> # Reorder to ZYX
            >>> permuted = space.apply_permute([2, 1, 0])
            >>> print(permuted.shape)
            (50, 200, 100)
            >>> print(permuted.spacing)
            (3.0, 2.0, 1.0)
        """
        assert sorted(axis_order) == [0, 1, 2], "axis_order must be a permutation of [0, 1, 2]"
        R = self._orientation_matrix()[:, axis_order]
        new_shape = tuple(np.array(self.shape)[axis_order])
        new_spacing = tuple(np.array(self.spacing)[axis_order])
        x_o, y_o, z_o = (R[:, 0], R[:, 1], R[:, 2])
        return Space(
            shape=new_shape,
            origin=self.origin,
            spacing=new_spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    def apply_bbox(self, bbox: np.ndarray, include_end: bool = False) -> "Space":
        """Crop the space to a bounding box.
        
        This method creates a new space that represents a cropped region of the
        original space, updating the origin and shape accordingly.
        
        Args:
            bbox: Bounding box array of shape (3, 2) where bbox[:,0] is start
                  indices and bbox[:,1] is end indices (exclusive)
            include_end: If True, include the end indices in the crop
            
        Returns:
            Space: New Space instance representing the cropped region
            
        Raises:
            AssertionError: If bbox shape is not (3, 2) or bounds are invalid
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> bbox = np.array([[10, 90], [20, 80], [5, 45]])
            >>> cropped = space.apply_bbox(bbox)
            >>> print(cropped.shape)
            (80, 60, 40)
            >>> print(cropped.origin)
            (10.0, 20.0, 10.0)
        """
        bbox = np.asarray(bbox)
        assert bbox.shape == (3, 2), "bbox must be a 3×2 array"
        assert np.all(bbox[:, 1] > bbox[:, 0]), "bbox upper bounds must be greater than lower bounds"
        # shift world origin
        shift = self._orientation_matrix() @ (bbox[:, 0] * np.array(self.spacing))
        new_origin = tuple(np.array(self.origin) + shift)
        new_shape = bbox[:, 1] - bbox[:, 0]
        if include_end:
            new_shape += 1
        return Space(
            shape=tuple(new_shape.tolist()),
            origin=new_origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_shape(self, shape: Tuple[int, int, int]) -> "Space":
        """Create a new space with modified shape only.
        
        This method creates a new space with the specified shape while
        preserving all other attributes (origin, spacing, orientation).
        
        Args:
            shape: New image dimensions (height, width, depth) in voxels
            
        Returns:
            Space: New Space instance with the specified shape
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> resized = space.apply_shape((200, 200, 100))
            >>> print(resized.shape)
            (200, 200, 100)
            >>> print(resized.spacing)  # Unchanged
            (1.0, 1.0, 2.0)
        """
        return Space(
            shape=shape,
            origin=self.origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_spacing(self, spacing: Tuple[float, float, float]) -> "Space":
        """Create a new space with modified spacing only.
        
        This method creates a new space with the specified voxel spacing while
        preserving all other attributes (origin, shape, orientation).
        
        Args:
            spacing: New voxel spacing (x, y, z) in mm
            
        Returns:
            Space: New Space instance with the specified spacing
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> resampled = space.apply_spacing((0.5, 0.5, 1.0))
            >>> print(resampled.spacing)
            (0.5, 0.5, 1.0)
            >>> print(resampled.shape)  # Unchanged
            (100, 100, 50)
        """
        return Space(
            shape=self.shape,
            origin=self.origin,
            spacing=spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_float_bbox(self, bbox: np.ndarray, shape: Tuple[int, int, int]) -> "Space":
        """Crop with floating-point bounding box and resample to specified shape.
        
        This method performs a floating-point crop followed by resampling to the
        target shape. The bbox coordinates can be non-integer values, allowing
        for sub-voxel precision cropping.
        
        Design Philosophy:
            Combines cropping and resampling in a single operation to avoid
            accumulation of interpolation errors. The floating-point bbox allows
            for precise sub-voxel alignment in medical image registration.
        
        Args:
            bbox: Bounding box array of shape (3, 2) where bbox[:,0] is start
                  coordinates (can be float) and bbox[:,1] is end coordinates
            shape: Target voxel dimensions (int, int, int) after resampling
            
        Returns:
            Space: New Space instance with specified shape and physical region
                  defined by the bounding box
            
        Raises:
            AssertionError: If bbox shape is not (3, 2) or bounds are invalid
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> bbox = np.array([[10.5, 90.5], [20.2, 80.8], [5.1, 45.9]])
            >>> resampled = space.apply_float_bbox(bbox, (80, 60, 40))
            >>> print(resampled.shape)
            (80, 60, 40)
            >>> print(resampled.spacing)
            (1.0, 1.0, 1.0)
        """
        # Input validation
        bbox = np.asarray(bbox, dtype=float)
        assert bbox.shape == (3, 2), "bbox must be a 3×2 array"
        assert np.all(bbox[:, 1] >= bbox[:, 0]), "bbox[:,1] must be >= bbox[:,0]"

        new_shape = np.asarray(shape, dtype=int)
        assert new_shape.shape == (3,), "shape must have length 3"
        assert np.all(new_shape > 0), "shape dimensions must be > 0"

        R = self._orientation_matrix()  # 3×3 column vectors

        # 1) New origin
        shift = R @ (bbox[:, 0] * np.array(self.spacing))  # world coordinate shift
        new_origin = np.array(self.origin) + shift

        # 2) New spacing
        physical_span = np.array(self.spacing) * (bbox[:, 1] - bbox[:, 0])
        tmp = new_shape.astype(float) - 1.0
        tmp[tmp == 0] = 1e-3  # avoid division by zero
        new_spacing = physical_span / tmp

        # Special handling for singular axes (target length = 1)
        singular_axis = new_shape == 1
        if np.any(singular_axis):
            new_spacing[singular_axis] = (
                np.array(self.spacing) * np.array(self.shape) / new_shape
            )[singular_axis]
            # Adjust origin to maintain physical center consistency
            shift2 = R @ (np.array(self.spacing) * (bbox[:, 1] - bbox[:, 0]))
            new_origin[singular_axis] = (new_origin + shift2 / 2)[singular_axis]

        return Space(
            shape=tuple(new_shape.tolist()),
            origin=tuple(new_origin.tolist()),
            spacing=tuple(new_spacing.tolist()),
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_zoom(
        self,
        factor: Union[Tuple[float, float, float], List[float], np.ndarray, float],
        *,
        mode: str = "floor",
        align_corners: bool = True,
    ) -> "Space":
        """Scale the shape by the given factor.
        
        This method scales the image dimensions by the specified factor(s) while
        keeping spacing and orientation unchanged. The mode parameter controls
        how non-integer results are handled.
        
        Args:
            factor: Scaling factor(s). Can be a single float or tuple of 3 floats
            mode: Rounding mode for non-integer results ("floor", "round", "ceil")
            align_corners: Currently ignored for backward compatibility
            
        Returns:
            Space: New Space instance with scaled shape
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> zoomed = space.apply_zoom(0.5, mode="round")
            >>> print(zoomed.shape)
            (50, 50, 25)
            >>> print(zoomed.spacing)  # Unchanged
            (1.0, 1.0, 2.0)
        """
        assert mode in {"floor", "round", "ceil"}, "mode must be floor/round/ceil"
        if np.isscalar(factor):
            factor = (float(factor),) * 3
        factor_arr = np.asarray(factor, dtype=float)
        assert factor_arr.shape == (3,), "factor must be length 3 or scalar"

        if mode == "floor":
            new_shape = np.floor(np.array(self.shape) * factor_arr).astype(int)
        elif mode == "round":
            new_shape = np.round(np.array(self.shape) * factor_arr).astype(int)
        else:  # ceil
            new_shape = np.ceil(np.array(self.shape) * factor_arr).astype(int)

        # Ensure minimum size of 1
        new_shape[new_shape < 1] = 1

        return self.apply_shape(tuple(new_shape.tolist()))

    # ------------------------------------------------------------------
    # Rotation methods
    # ------------------------------------------------------------------
    def _axis_angle_rotation(self, axis: int, angle_rad: float) -> np.ndarray:
        """Generate rotation matrix for rotation around a coordinate axis.
        
        This internal method generates a 3x3 rotation matrix for rotation
        around one of the coordinate axes using Rodrigues' rotation formula.
        
        Args:
            axis: Axis of rotation (0=X, 1=Y, 2=Z)
            angle_rad: Rotation angle in radians
            
        Returns:
            np.ndarray: 3x3 rotation matrix
            
        Raises:
            ValueError: If axis is not 0, 1, or 2
        """
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        if axis == 0:  # X axis
            rot = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=float,
            )
        elif axis == 1:  # Y axis
            rot = np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=float,
            )
        elif axis == 2:  # Z axis
            rot = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=float,
            )
        else:
            raise ValueError("axis should be 0/1/2")
        return rot

    def apply_rotate(
        self,
        axis: int,
        angle: float,
        *,
        unit: str = "radian",
        center: str = "center",
    ) -> "Space":
        """Rotate the space around a specified axis.
        
        This method rotates the image space around one of the coordinate axes,
        updating the orientation vectors and optionally the origin.
        
        Args:
            axis: Axis of rotation (0=x, 1=y, 2=z)
            angle: Rotation angle
            unit: Angle unit ("radian" or "degree")
            center: Rotation center ("center" for image center, "origin" for world origin)
            
        Returns:
            Space: New Space instance with rotated orientation
            
        Raises:
            AssertionError: If parameters are invalid
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> rotated = space.apply_rotate(2, 90, unit="degree", center="center")
            >>> print(rotated.x_orientation)  # Rotated 90 degrees around z
            (0.0, 1.0, 0.0)
            >>> print(rotated.y_orientation)
            (-1.0, 0.0, 0.0)
        """
        assert axis in (0, 1, 2), "axis must be 0/1/2"
        assert unit in ("radian", "degree"), "unit must be radian/degree"
        assert center in ("center", "origin"), "center must be center/origin"

        angle_rad = float(angle) if unit == "radian" else float(angle) / 180.0 * np.pi

        R_old = self._orientation_matrix()  # 3×3
        rotm = self._axis_angle_rotation(axis, angle_rad)  # 3×3
        R_new = R_old @ rotm  # right multiply column vectors

        # Split column vectors
        x_o, y_o, z_o = (R_new[:, 0], R_new[:, 1], R_new[:, 2])

        if center == "center":
            center_world = np.array(self.origin) + self.physical_span / 2.0
            extent = (np.array(self.shape) - 1) * np.array(self.spacing)
            new_span = R_new @ extent
            new_origin = center_world - new_span / 2.0
        else:
            new_origin = np.array(self.origin)

        return Space(
            shape=self.shape,
            origin=tuple(new_origin.tolist()),
            spacing=self.spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def copy(self) -> "Space":
        """Return a new Space instance with identical values.
        
        Creates a deep copy of the Space object with all the same attribute values.
        
        Returns:
            Space: New Space instance with identical values
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> space_copy = space.copy()
            >>> print(space_copy == space)  # Same values
            True
            >>> print(space_copy is space)  # Different objects
            False
        """
        return Space(
            shape=self.shape,
            origin=self.origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def contain_pointset_ind(self, pointset_ind: np.ndarray) -> np.ndarray:
        """Check if index coordinates are within the space bounds.
        
        This method tests whether the given index coordinates fall within
        the valid range [0, shape-1] for each dimension.
        
        Args:
            pointset_ind: Array of index coordinates with shape (N, 3)
            
        Returns:
            np.ndarray: Boolean array of shape (N,) indicating which points are inside
            
        Example:
            >>> space = Space(shape=(100, 100, 50))
            >>> points = np.array([[10, 20, 30], [150, 50, 25], [50, 50, 25]])
            >>> inside = space.contain_pointset_ind(points)
            >>> print(inside)
            [True False True]
        """
        pts = np.asarray(pointset_ind)
        assert pts.ndim == 2 and pts.shape[1] == 3
        return np.all((pts >= 0) & (pts <= np.array(self.shape)[None] - 1), axis=1)

    def contain_pointset_world(self, pointset_world: np.ndarray) -> np.ndarray:
        """Check if world coordinates are within the space bounds.
        
        This method converts world coordinates to index coordinates and then
        checks if they fall within the valid index range.
        
        Args:
            pointset_world: Array of world coordinates with shape (N, 3)
            
        Returns:
            np.ndarray: Boolean array of shape (N,) indicating which points are inside
            
        Example:
            >>> space = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
            >>> world_points = np.array([[10.0, 20.0, 30.0], [150.0, 50.0, 25.0]])
            >>> inside = space.contain_pointset_world(world_points)
            >>> print(inside)
            [True False]
        """
        pts_idx = self.from_world_transform.apply_piont(pointset_world)
        return self.contain_pointset_ind(pts_idx)


def get_space_from_nifti(niftiimage: "NiftiImage") -> "Space":
    """Create a Space object from a NIfTI image.
    
    This function extracts geometric information from a NIfTI image including
    orientation, spacing, and origin information from the affine matrix.
    
    Args:
        niftiimage: NIfTI image object with affine matrix and shape
        
    Returns:
        Space: A new Space instance with geometry matching the NIfTI image
        
    Raises:
        ValueError: If affine matrix is not 4x4
        
    Example:
        >>> import nibabel as nib
        >>> image = nib.load('brain.nii.gz')
        >>> space = get_space_from_nifti(image)
        >>> print(space.shape)
        (256, 256, 256)
        >>> print(space.spacing)
        (1.0, 1.0, 1.0)
    """
    affine = niftiimage.affine
    shape = niftiimage.shape

    if affine.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4.")

    # Extract direction cosines and spacing
    R = affine[:3, :3]  # Extract rotation part
    spacing = np.linalg.norm(R, axis=0)  # Spacing is column norms
    orientation = R / spacing  # Normalize direction cosines

    origin = tuple(affine[:3, 3].tolist())  # Extract origin
    x_orientation = tuple(orientation[:, 0].tolist())
    y_orientation = tuple(orientation[:, 1].tolist())
    z_orientation = tuple(orientation[:, 2].tolist())

    return Space(
        origin=origin,
        spacing=tuple(spacing.tolist()),
        x_orientation=x_orientation,
        y_orientation=y_orientation,
        z_orientation=z_orientation,
        shape=shape,
    )


def get_space_from_sitk(simpleitkimage) -> "Space":
    """Create a Space object from a SimpleITK Image.
    
    This function extracts geometric information from a SimpleITK Image including
    origin coordinates, voxel spacing, direction cosines, and image dimensions.
    
    Args:
        simpleitkimage: SimpleITK Image object
        
    Returns:
        Space: A new Space instance with geometry matching the SimpleITK image
        
    Example:
        >>> import SimpleITK as sitk
        >>> image = sitk.ReadImage('brain.nii.gz')
        >>> space = get_space_from_sitk(image)
        >>> print(space.shape)
        (256, 256, 256)
        >>> print(space.spacing)
        (1.0, 1.0, 1.0)
    """
    origin = simpleitkimage.GetOrigin()
    spacing = simpleitkimage.GetSpacing()
    d = simpleitkimage.GetDirection()  # Flattened in column-major order
    size = simpleitkimage.GetSize()

    # Extract direction vectors from flattened matrix
    x_orientation = [d[0], d[3], d[6]]
    y_orientation = [d[1], d[4], d[7]]
    z_orientation = [d[2], d[5], d[8]]
    shape = size

    return Space(
        origin=origin,
        spacing=spacing,
        x_orientation=x_orientation,
        y_orientation=y_orientation,
        z_orientation=z_orientation,
        shape=shape,
    )

