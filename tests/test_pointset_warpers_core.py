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
        warp2 = T.apply_points(pts)
        self.assertTrue(np.allclose(warp1, warp2, atol=1e-5))

    def test_warp_vector(self):
        s = self._random_space()
        t = self._random_space()
        vecs = np.random.randn(40, 3)
        T = calc_transform(s, t)
        warp1 = warp_vector(vecs, s, t)
        warp2 = T.apply_vectors(vecs)
        self.assertTrue(np.allclose(warp1, warp2, atol=1e-5))


if __name__ == "__main__":
    unittest.main() 