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
        out1 = t3.apply_points(pts)
        out2 = t2.apply_points(t1.apply_points(pts))
        self.assertTrue(np.allclose(out1, out2, atol=1e-6))
        # compose 同等语义
        t3b = t1.compose(t2)  # 先 t1 后 t2
        self.assertTrue(np.allclose(t3b.matrix, t3.matrix))

    def test_apply_vectors(self):
        mat = self._random_matrix()
        T = Transform(mat)
        vecs = np.random.rand(8, 3)
        out = T.apply_vectors(vecs)
        # 对向量应用，与直接乘以旋转部分等价
        expected = (mat[:3, :3] @ vecs.T).T
        self.assertTrue(np.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main() 