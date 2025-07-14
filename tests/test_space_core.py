import os
import sys
import unittest
import numpy as np

# 确保优先加载新实现
CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.space import Space


class TestSpace(unittest.TestCase):
    def setUp(self):
        shape = (8, 9, 10)
        spacing = (0.9, 0.8, 1.0)
        # 生成随机正交方向余弦
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        self.xo, self.yo, self.zo = R[:, 0], R[:, 1], R[:, 2]
        self.origin = (0.1, 0.2, 0.3)
        self.sp = Space(
            shape=shape,
            origin=self.origin,
            spacing=spacing,
            x_orientation=tuple(self.xo),
            y_orientation=tuple(self.yo),
            z_orientation=tuple(self.zo),
        )

    def test_world_index_roundtrip(self):
        idx = np.random.rand(20, 3) * (np.array(self.sp.shape) - 1)
        world = self.sp.to_world_transform.apply_piont(idx)
        idx_back = self.sp.from_world_transform.apply_piont(world)
        self.assertTrue(np.allclose(idx, idx_back, atol=1e-5))

    def test_physical_span_end(self):
        span = self.sp.physical_span
        manual_span = (
            (np.array(self.sp.shape) - 1)
            * np.array(self.sp.spacing)
        )
        manual_span_vec = (
            self.xo * manual_span[0]
            + self.yo * manual_span[1]
            + self.zo * manual_span[2]
        )
        self.assertTrue(np.allclose(span, manual_span_vec, atol=1e-6))
        end = self.sp.end
        self.assertTrue(np.allclose(end, np.array(self.origin) + manual_span_vec, atol=1e-6))

    def test_apply_flip(self):
        sp2 = self.sp.apply_flip(0)
        # x 向量应取反
        self.assertTrue(np.allclose(sp2.x_orientation, -np.array(self.sp.x_orientation)))
        self.assertTrue(np.allclose(sp2.y_orientation, np.array(self.sp.y_orientation)))
        # origin.x 应移动到 end.x
        self.assertAlmostEqual(sp2.origin[0], self.sp.end[0])

    def test_apply_permute(self):
        order = [2, 0, 1]
        sp2 = self.sp.apply_permute(order)
        self.assertEqual(sp2.shape, tuple(np.array(self.sp.shape)[order]))
        self.assertEqual(sp2.spacing, tuple(np.array(self.sp.spacing)[order]))
        # orientation 列应该重新排列
        R2 = np.column_stack((sp2.x_orientation, sp2.y_orientation, sp2.z_orientation))
        R = np.column_stack((self.sp.x_orientation, self.sp.y_orientation, self.sp.z_orientation))
        self.assertTrue(np.allclose(R2, R[:, order]))

    def test_apply_bbox(self):
        bbox = np.array([[1, 6], [2, 8], [0, 9]])
        sp2 = self.sp.apply_bbox(bbox)
        expected_shape = tuple((bbox[:, 1] - bbox[:, 0]).tolist())
        self.assertEqual(sp2.shape, expected_shape)
        # world origin 应该偏移
        shift = (
            self.xo * (bbox[0, 0] * self.sp.spacing[0])
            + self.yo * (bbox[1, 0] * self.sp.spacing[1])
            + self.zo * (bbox[2, 0] * self.sp.spacing[2])
        )
        self.assertTrue(np.allclose(np.array(sp2.origin), np.array(self.origin) + shift))

    def test_apply_float_bbox(self):
        """测试浮点 bbox 裁剪后 shape / spacing / origin 是否正确。"""
        bbox = np.array([[1.2, 5.6], [0.5, 7.3], [2.0, 9.0]], dtype=float)
        new_shape = (11, 15, 5)
        sp2 = self.sp.apply_float_bbox(bbox, new_shape)
        # shape 应一致
        self.assertEqual(sp2.shape, new_shape)
        # spacing 计算
        span_phys = np.array(self.sp.spacing) * (bbox[:, 1] - bbox[:, 0])
        expected_spacing = span_phys / (np.array(new_shape) - 1)
        # 若 shape 轴为 1，则 spacing 直接继承原实现中的特殊处理，无法简单验证，故只验证非 1 轴
        mask = np.array(new_shape) != 1
        self.assertTrue(np.allclose(np.array(sp2.spacing)[mask], expected_spacing[mask], atol=1e-6))
        # origin 偏移
        shift = (
            self.xo * (bbox[0, 0] * self.sp.spacing[0])
            + self.yo * (bbox[1, 0] * self.sp.spacing[1])
            + self.zo * (bbox[2, 0] * self.sp.spacing[2])
        )
        self.assertTrue(np.allclose(np.array(sp2.origin), np.array(self.origin) + shift, atol=1e-6))

    def test_apply_zoom(self):
        factor = (0.5, 2.0, 1.3)
        sp2 = self.sp.apply_zoom(factor, mode="round")
        expected_shape = tuple(np.round(np.array(self.sp.shape) * np.array(factor)).astype(int).tolist())
        self.assertEqual(sp2.shape, expected_shape)
        # spacing 不变
        self.assertEqual(sp2.spacing, self.sp.spacing)
        # origin 不变
        self.assertEqual(sp2.origin, self.sp.origin)

    def test_apply_rotate(self):
        axis = 2  # z 轴
        angle_deg = 90
        sp2 = self.sp.apply_rotate(axis, angle_deg, unit="degree", center="origin")

        # 构造期望方向矩阵
        c, s = 0, 1  # cos90, sin90
        rotm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R_old = np.column_stack((self.xo, self.yo, self.zo))
        R_expected = R_old @ rotm
        R_new = np.column_stack((sp2.x_orientation, sp2.y_orientation, sp2.z_orientation))
        self.assertTrue(np.allclose(R_new, R_expected, atol=1e-6))
        # origin 应保持不变
        self.assertEqual(sp2.origin, self.sp.origin)


if __name__ == "__main__":
    unittest.main() 