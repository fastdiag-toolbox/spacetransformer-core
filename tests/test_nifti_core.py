import os
import sys
import unittest
import numpy as np

# 确保优先加载新实现
CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.space import Space, get_space_from_nifti, get_space_from_sitk


class TestNiftiCore(unittest.TestCase):
    """Test NIfTI-related core functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Test file path
        self.test_nifti_file = os.path.join(CORE_ROOT, "example", "CT_Philips.nii.gz")
        
        # Check dependencies
        try:
            import nibabel as nib
            self.nib = nib
        except ImportError:
            self.skipTest("nibabel not available")
            
        try:
            import SimpleITK as sitk
            self.sitk = sitk
        except ImportError:
            self.skipTest("SimpleITK not available")
        
        # Check test file
        if not os.path.exists(self.test_nifti_file):
            self.skipTest(f"Test file not found: {self.test_nifti_file}")
            
        # Load test data using SimpleITK as reference
        sitk_img = self.sitk.ReadImage(self.test_nifti_file)
        self.space = get_space_from_sitk(sitk_img)
    
    def test_get_space_from_nifti_basic(self):
        """Test basic reading from NIfTI file"""
        nib_img = self.nib.load(self.test_nifti_file)
        space = get_space_from_nifti(nib_img)
        
        # Verify basic attributes
        self.assertIsInstance(space, Space)
        self.assertEqual(len(space.shape), 3)
        self.assertEqual(len(space.origin), 3)
        self.assertEqual(len(space.spacing), 3)
        
        # Verify positive values
        self.assertTrue(all(s > 0 for s in space.shape))
        self.assertTrue(all(sp > 0 for sp in space.spacing))
        
        # Verify unit length orientation vectors
        x_norm = np.linalg.norm(space.x_orientation)
        y_norm = np.linalg.norm(space.y_orientation)
        z_norm = np.linalg.norm(space.z_orientation)
        self.assertAlmostEqual(x_norm, 1.0, places=6)
        self.assertAlmostEqual(y_norm, 1.0, places=6)
        self.assertAlmostEqual(z_norm, 1.0, places=6)
    
    def test_qform_priority_over_sform(self):
        """Test qform takes priority over sform"""
        nib_img = self.nib.load(self.test_nifti_file)
        
        # Get qform and sform
        qform_affine, qform_code = nib_img.get_qform(coded=True)
        sform_affine, sform_code = nib_img.get_sform(coded=True)
        
        # If qform is valid, test it's used
        if qform_code > 0:
            space = get_space_from_nifti(nib_img)
            generated_affine = space.to_nifti_affine()
            
            # Should match qform, not sform
            self.assertTrue(np.allclose(generated_affine, qform_affine, atol=1e-6))
    
    def test_coordinate_system_lps(self):
        """Test LPS coordinate system conversion"""
        nib_img = self.nib.load(self.test_nifti_file)
        space = get_space_from_nifti(nib_img)
        
        # For standard axial images, expect LPS orientation
        expected_x = (-1.0, 0.0, 0.0)  # Left
        expected_y = (0.0, -1.0, 0.0)  # Posterior
        expected_z = (0.0, 0.0, 1.0)   # Superior
        
        self.assertTrue(np.allclose(space.x_orientation, expected_x, atol=1e-6))
        self.assertTrue(np.allclose(space.y_orientation, expected_y, atol=1e-6))
        self.assertTrue(np.allclose(space.z_orientation, expected_z, atol=1e-6))
    
    def test_nifti_roundtrip_consistency(self):
        """Test roundtrip consistency: NIfTI -> Space -> NIfTI"""
        nib_img = self.nib.load(self.test_nifti_file)
        qform_affine, qform_code = nib_img.get_qform(coded=True)
        
        if qform_code > 0:
            space = get_space_from_nifti(nib_img)
            roundtrip_affine = space.to_nifti_affine()
            
            # Should be identical
            self.assertTrue(np.allclose(qform_affine, roundtrip_affine, atol=1e-6))
    
    def test_coordinate_transform_accuracy(self):
        """Test coordinate transformation accuracy"""
        # Test several points
        shape = np.array(self.space.shape)
        test_indices = np.array([
            [0, 0, 0],
            shape // 2,
            shape - 1,
        ], dtype=float)
        
        # Test roundtrip: index -> world -> index
        world_coords = self.space.to_world_transform.apply_point(test_indices)
        back_to_indices = self.space.from_world_transform.apply_point(world_coords)
        
        # Verify roundtrip accuracy
        max_diff = np.max(np.abs(test_indices - back_to_indices))
        self.assertLess(max_diff, 1e-5)
    
    def test_consistency_with_simpleitk(self):
        """Test consistency between nibabel and SimpleITK reads"""
        nib_img = self.nib.load(self.test_nifti_file)
        space_from_nib = get_space_from_nifti(nib_img)
        
        # Should match our reference space (from SimpleITK)
        self.assertEqual(space_from_nib.shape, self.space.shape)
        
        # Check geometric properties
        origin_diff = np.max(np.abs(np.array(space_from_nib.origin) - np.array(self.space.origin)))
        spacing_diff = np.max(np.abs(np.array(space_from_nib.spacing) - np.array(self.space.spacing)))
        
        self.assertLess(origin_diff, 1e-6)
        self.assertLess(spacing_diff, 1e-6)
        
        # Check orientation vectors
        x_diff = np.max(np.abs(np.array(space_from_nib.x_orientation) - np.array(self.space.x_orientation)))
        y_diff = np.max(np.abs(np.array(space_from_nib.y_orientation) - np.array(self.space.y_orientation)))
        z_diff = np.max(np.abs(np.array(space_from_nib.z_orientation) - np.array(self.space.z_orientation)))
        
        self.assertLess(x_diff, 1e-6)
        self.assertLess(y_diff, 1e-6)
        self.assertLess(z_diff, 1e-6)
    
    def test_random_points_roundtrip(self):
        """Test roundtrip with random points"""
        np.random.seed(42)
        shape = np.array(self.space.shape)
        n_points = 20
        
        # Generate random points within valid range
        random_indices = np.random.rand(n_points, 3) * (shape - 1)
        
        # Roundtrip transformation
        world_coords = self.space.to_world_transform.apply_point(random_indices)
        back_to_indices = self.space.from_world_transform.apply_point(world_coords)
        
        # Verify accuracy
        max_diff = np.max(np.abs(random_indices - back_to_indices))
        self.assertLess(max_diff, 1e-5)
    
    def test_shape_axis_order(self):
        """Test shape axis ordering"""
        nib_img = self.nib.load(self.test_nifti_file)
        space = get_space_from_nifti(nib_img)
        
        # Shape should match NIfTI
        self.assertEqual(space.shape, nib_img.shape)
        
        # shape_zyx should be reversed
        expected_zyx = nib_img.shape[::-1]
        self.assertEqual(space.shape_zyx, expected_zyx)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        class MockInvalidNiftiImage:
            def __init__(self):
                self.affine = np.eye(3)  # Wrong size
                self.shape = (100, 100, 50)
                
            def get_qform(self, coded=False):
                return self.affine, 0
        
        mock_img = MockInvalidNiftiImage()
        with self.assertRaises(ValueError):
            get_space_from_nifti(mock_img)


if __name__ == '__main__':
    unittest.main(verbosity=2)