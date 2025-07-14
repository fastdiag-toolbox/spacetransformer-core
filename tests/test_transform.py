import os
import sys
import unittest
import numpy as np

# 确保优先加载新实现
CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.transform import Transform


class TestTransform(unittest.TestCase):
    def _random_matrix(self) -> np.ndarray:
        """生成随机可逆 4×4 齐次矩阵。"""
        # 随机正交矩阵 + 缩放
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        S = np.diag(np.random.rand(3) + 0.2)
        RS = R @ S
        t = np.random.rand(3) * 10
        mat = np.eye(4)
        mat[:3, :3] = RS
        mat[:3, 3] = t
        return mat

    def test_inverse(self):
        for _ in range(10):
            mat = self._random_matrix()
            T = Transform(mat)
            Tinv = T.inverse()
            self.assertTrue(np.allclose(Tinv.matrix @ T.matrix, np.eye(4), atol=1e-6))
            # cache 是否复用
            self.assertIs(T.inverse(), Tinv)

    def test_compose(self):
        mat1 = self._random_matrix()
        mat2 = self._random_matrix()
        t1 = Transform(mat1)
        t2 = Transform(mat2)
        t3 = t2 @ t1  # 先 t1 后 t2
        pts = np.random.rand(5, 3)
        out1 = t3.apply_piont(pts)
        out2 = t2.apply_piont(t1.apply_piont(pts))
        self.assertTrue(np.allclose(out1, out2, atol=1e-6))
        # compose 同等语义
        t3b = t1.compose(t2)  # 先 t1 后 t2
        self.assertTrue(np.allclose(t3b.matrix, t3.matrix))

    def test_apply_vector(self):
        mat = self._random_matrix()
        T = Transform(mat)
        vecs = np.random.rand(8, 3)
        out = T.apply_vector(vecs)
        # 对向量应用，与直接乘以旋转部分等价
        expected = (mat[:3, :3] @ vecs.T).T
        self.assertTrue(np.allclose(out, expected, atol=1e-6))

    def test_apply_piont_list_tuple_input(self):
        """测试 apply_piont 对 list 和 tuple 输入的处理"""
        mat = self._random_matrix()
        T = Transform(mat)
        
        # 测试单个点 (list)
        point_list = [1.0, 2.0, 3.0]
        result_list = T.apply_piont(point_list)
        self.assertEqual(result_list.shape, (1, 3))
        
        # 测试单个点 (tuple)
        point_tuple = (1.0, 2.0, 3.0)
        result_tuple = T.apply_piont(point_tuple)
        self.assertEqual(result_tuple.shape, (1, 3))
        
        # 验证 list 和 tuple 结果一致
        self.assertTrue(np.allclose(result_list, result_tuple, atol=1e-6))
        
        # 测试与 numpy 数组结果一致
        point_array = np.array([1.0, 2.0, 3.0])
        result_array = T.apply_piont(point_array)
        self.assertTrue(np.allclose(result_list, result_array, atol=1e-6))
        
        # 测试多个点 (list of lists)
        points_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result_multi = T.apply_piont(points_list)
        self.assertEqual(result_multi.shape, (2, 3))
        
        # 验证与 numpy 数组结果一致
        points_array = np.array(points_list)
        result_multi_array = T.apply_piont(points_array)
        self.assertTrue(np.allclose(result_multi, result_multi_array, atol=1e-6))

    def test_apply_vector_list_tuple_input(self):
        """测试 apply_vector 对 list 和 tuple 输入的处理"""
        mat = self._random_matrix()
        T = Transform(mat)
        
        # 测试单个向量 (list)
        vec_list = [1.0, 0.0, 0.0]
        result_list = T.apply_vector(vec_list)
        self.assertEqual(result_list.shape, (1, 3))
        
        # 测试单个向量 (tuple)
        vec_tuple = (1.0, 0.0, 0.0)
        result_tuple = T.apply_vector(vec_tuple)
        self.assertEqual(result_tuple.shape, (1, 3))
        
        # 验证 list 和 tuple 结果一致
        self.assertTrue(np.allclose(result_list, result_tuple, atol=1e-6))
        
        # 测试与 numpy 数组结果一致
        vec_array = np.array([1.0, 0.0, 0.0])
        result_array = T.apply_vector(vec_array)
        self.assertTrue(np.allclose(result_list, result_array, atol=1e-6))
        
        # 测试多个向量 (list of lists)
        vecs_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result_multi = T.apply_vector(vecs_list)
        self.assertEqual(result_multi.shape, (2, 3))
        
        # 验证与 numpy 数组结果一致
        vecs_array = np.array(vecs_list)
        result_multi_array = T.apply_vector(vecs_array)
        self.assertTrue(np.allclose(result_multi, result_multi_array, atol=1e-6))

    def test_call_method_list_tuple_input(self):
        """测试 __call__ 方法对 list 和 tuple 输入的处理"""
        mat = self._random_matrix()
        T = Transform(mat)
        
        # 测试 __call__ 方法与 apply_piont 的一致性
        point_list = [1.0, 2.0, 3.0]
        result_call = T(point_list)
        result_apply = T.apply_piont(point_list)
        self.assertTrue(np.allclose(result_call, result_apply, atol=1e-6))


if __name__ == "__main__":
    unittest.main() 