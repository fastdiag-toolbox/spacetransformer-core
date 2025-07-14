import os
import sys
import unittest
import numpy as np

CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.space import Space
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
        warp2 = T.apply_piont(pts)
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


if __name__ == "__main__":
    unittest.main() 