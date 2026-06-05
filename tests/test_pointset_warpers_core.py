import os
import sys
import unittest
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.space import Space
import spacetransformer.core.pointset_warpers as pointset_warpers
from spacetransformer.core.pointset_warpers import calc_transform, warp_point, warp_vector


class TestPointWarpers(unittest.TestCase):
    def _random_space(self):
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        origin = tuple(np.random.rand(3))
        spacing = tuple(np.random.rand(3) + 0.3)
        shape = tuple(np.random.randint(5, 20, size=3))
        return Space(
            shape=shape,
            origin=origin,
            spacing=spacing,
            x_orientation=tuple(R[:, 0]),
            y_orientation=tuple(R[:, 1]),
            z_orientation=tuple(R[:, 2]),
        )

    def test_calc_theta(self):
        s = self._random_space()
        t = self._random_space()
        T = calc_transform(s, t)
        pts = np.random.rand(30, 3) * (np.array(s.shape) - 1)
        warp1, _ = warp_point(pts, s, t)
        warp2 = T.apply_point(pts)
        self.assertTrue(np.allclose(warp1, warp2, atol=1e-5))

    def test_warp_vector(self):
        s = self._random_space()
        t = self._random_space()
        vecs = np.random.randn(40, 3)
        T = calc_transform(s, t)
        warp1 = warp_vector(vecs, s, t)
        warp2 = T.apply_vector(vecs)
        self.assertTrue(np.allclose(warp1, warp2, atol=1e-5))

    def test_warp_point_list_tuple_input(self):
        """测试 warp_point 对 list 和 tuple 输入的处理"""
        s = self._random_space()
        t = self._random_space()
        
        # 测试单个点 (list)
        point_list = [1.0, 2.0, 3.0]
        warp_list, isin_list = warp_point(point_list, s, t)
        self.assertEqual(warp_list.shape, (1, 3))
        self.assertEqual(isin_list.shape, (1,))
        
        # 测试单个点 (tuple)
        point_tuple = (1.0, 2.0, 3.0)
        warp_tuple, isin_tuple = warp_point(point_tuple, s, t)
        self.assertEqual(warp_tuple.shape, (1, 3))
        self.assertEqual(isin_tuple.shape, (1,))
        
        # 验证 list 和 tuple 结果一致
        self.assertTrue(np.allclose(warp_list, warp_tuple, atol=1e-6))
        self.assertTrue(np.allclose(isin_list, isin_tuple))
        
        # 测试与 numpy 数组结果一致
        point_array = np.array([1.0, 2.0, 3.0])
        warp_array, isin_array = warp_point(point_array, s, t)
        self.assertTrue(np.allclose(warp_list, warp_array, atol=1e-6))
        self.assertTrue(np.allclose(isin_list, isin_array))
        
        # 测试多个点 (list of lists)
        points_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        warp_multi, isin_multi = warp_point(points_list, s, t)
        self.assertEqual(warp_multi.shape, (2, 3))
        self.assertEqual(isin_multi.shape, (2,))
        
        # 验证与 numpy 数组结果一致
        points_array = np.array(points_list)
        warp_multi_array, isin_multi_array = warp_point(points_array, s, t)
        self.assertTrue(np.allclose(warp_multi, warp_multi_array, atol=1e-6))
        self.assertTrue(np.allclose(isin_multi, isin_multi_array))

    def test_warp_vector_list_tuple_input(self):
        """测试 warp_vector 对 list 和 tuple 输入的处理"""
        s = self._random_space()
        t = self._random_space()
        
        # 测试单个向量 (list)
        vec_list = [1.0, 0.0, 0.0]
        warp_list = warp_vector(vec_list, s, t)
        self.assertEqual(warp_list.shape, (1, 3))
        
        # 测试单个向量 (tuple)
        vec_tuple = (1.0, 0.0, 0.0)
        warp_tuple = warp_vector(vec_tuple, s, t)
        self.assertEqual(warp_tuple.shape, (1, 3))
        
        # 验证 list 和 tuple 结果一致
        self.assertTrue(np.allclose(warp_list, warp_tuple, atol=1e-6))
        
        # 测试与 numpy 数组结果一致
        vec_array = np.array([1.0, 0.0, 0.0])
        warp_array = warp_vector(vec_array, s, t)
        self.assertTrue(np.allclose(warp_list, warp_array, atol=1e-6))
        
        # 测试多个向量 (list of lists)
        vecs_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        warp_multi = warp_vector(vecs_list, s, t)
        self.assertEqual(warp_multi.shape, (2, 3))
        
        # 验证与 numpy 数组结果一致
        vecs_array = np.array(vecs_list)
        warp_multi_array = warp_vector(vecs_array, s, t)
        self.assertTrue(np.allclose(warp_multi, warp_multi_array, atol=1e-6))

    def test_module_does_not_import_torch_eagerly(self):
        self.assertNotIn("torch", pointset_warpers.__dict__)

    def test_warp_point_torch_tensor_matches_numpy_without_cpu_roundtrip(self):
        torch = pytest.importorskip("torch")
        s = self._random_space()
        t = self._random_space()
        points_np = np.random.rand(128, 3).astype(np.float32) * (np.asarray(s.shape, dtype=np.float32) - 1)
        points = torch.as_tensor(points_np)

        warp_torch, isin_torch = warp_point(points, s, t)
        warp_np, isin_np = warp_point(points_np, s, t)

        self.assertIsInstance(warp_torch, torch.Tensor)
        self.assertIsInstance(isin_torch, torch.Tensor)
        self.assertEqual(warp_torch.device, points.device)
        torch.testing.assert_close(warp_torch.cpu(), torch.as_tensor(warp_np, dtype=warp_torch.dtype), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(isin_torch.cpu().numpy(), isin_np)

    def test_warp_vector_torch_tensor_matches_numpy_without_cpu_roundtrip(self):
        torch = pytest.importorskip("torch")
        s = self._random_space()
        t = self._random_space()
        vecs_np = np.random.randn(128, 3).astype(np.float32)
        vecs = torch.as_tensor(vecs_np)

        warp_torch = warp_vector(vecs, s, t)
        warp_np = warp_vector(vecs_np, s, t)

        self.assertIsInstance(warp_torch, torch.Tensor)
        self.assertEqual(warp_torch.device, vecs.device)
        torch.testing.assert_close(warp_torch.cpu(), torch.as_tensor(warp_np, dtype=warp_torch.dtype), rtol=1e-5, atol=1e-5)

    def test_warp_point_preserves_cuda_device(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        s = self._random_space()
        t = self._random_space()
        points = torch.rand((128, 3), dtype=torch.float32, device="cuda")

        warp_torch, isin_torch = warp_point(points, s, t)

        self.assertEqual(warp_torch.device.type, "cuda")
        self.assertEqual(isin_torch.device.type, "cuda")

    def test_warp_vector_preserves_cuda_device(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        s = self._random_space()
        t = self._random_space()
        vectors = torch.rand((128, 3), dtype=torch.float32, device="cuda")

        warp_torch = warp_vector(vectors, s, t)

        self.assertEqual(warp_torch.device.type, "cuda")


if __name__ == "__main__":
    unittest.main() 
