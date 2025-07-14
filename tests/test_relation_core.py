import os
import sys
import unittest
import numpy as np

CORE_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from spacetransformer.core.space import Space
from spacetransformer.core.relation_check import (
    _check_valid_flip_permute,
    _check_same_base,
    _check_same_spacing,
    _check_align_corner,
    _check_isin,
)


class TestRelationCheck(unittest.TestCase):
    def setUp(self):
        base = np.array([[0.8, 0.6, 0.0], [-0.6, 0.8, 0.6], [0.0, 0.0, 0.8]])
        self.shape = (8, 9, 10)
        self.sp1 = Space(
            shape=self.shape,
            origin=(0.1, 0.2, 0.3),
            spacing=(0.9, 0.8, 1.0),
            x_orientation=tuple(base[:, 0]),
            y_orientation=tuple(base[:, 1]),
            z_orientation=tuple(base[:, 2]),
        )

    def test_flip_permute(self):
        sp2 = self.sp1.apply_flip(1)
        flag, flips, order = _check_valid_flip_permute(self.sp1, sp2)
        self.assertTrue(flag)
        self.assertEqual(flips, [0, 1, 0])
        self.assertEqual(order, [0, 1, 2])

        sp3 = sp2.apply_flip(2)
        flag, flips, order = _check_valid_flip_permute(self.sp1, sp3)
        self.assertTrue(flag)
        self.assertEqual(flips, [0, 1, 1])
        self.assertEqual(order, [0, 1, 2])

        sp4 = sp3.apply_swap(0, 1)
        flag, flips, order = _check_valid_flip_permute(self.sp1, sp4)
        self.assertTrue(flag)
        self.assertEqual(order, [1, 0, 2])

    def test_same_base_spacing_align(self):
        # create target with same base spacing but cropped
        sp2 = self.sp1.apply_bbox(np.array([[1, 6], [2, 6], [0, 6]]))
        self.assertTrue(_check_same_base(self.sp1, sp2))
        self.assertTrue(_check_same_spacing(self.sp1, sp2))
        self.assertTrue(_check_align_corner(self.sp1, sp2))
        # alter spacing
        sp3 = sp2.apply_spacing((1.0, 1.0, 1.0))
        self.assertFalse(_check_same_spacing(self.sp1, sp3))

    def test_isin(self):
        sp2 = self.sp1.apply_bbox(np.array([[1, 6], [1, 8], [1, 9]]))
        self.assertTrue(_check_isin(sp2, self.sp1))
        sp3 = self.sp1.apply_spacing((2.0, 2.0, 2.0))
        self.assertFalse(_check_isin(sp3, self.sp1))


if __name__ == "__main__":
    unittest.main() 