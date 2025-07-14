from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 解决循环引用的类型提示
    from .space import Space


def _homogeneous(points: np.ndarray, w: float = 1.0) -> np.ndarray:
    """在最后一维追加常数 w 构成齐次坐标。"""
    if points.ndim == 1:
        points = points[None]
    ones = np.full((points.shape[0], 1), w, dtype=points.dtype)
    return np.concatenate([points, ones], axis=1)


@dataclass
class Transform:
    """4×4 齐次坐标变换矩阵封装。

    该类仅用于几何坐标计算，不涉及任何重采样相关参数。
    """

    matrix: np.ndarray  # 4×4 矩阵 (source.index → target.index 或其他)
    source: Optional["Space"] = None  # 源 Space，可为空（如 world）
    target: Optional["Space"] = None  # 目标 Space，可为空
    _inverse_cache: Optional["Transform"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.matrix.shape != (4, 4):
            raise ValueError("matrix 必须是 4×4 大小")

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------
    def inverse(self) -> "Transform":
        """返回自身的逆变换（懒计算并缓存）。"""
        if self._inverse_cache is None:
            inv_mat = np.linalg.inv(self.matrix)
            self._inverse_cache = Transform(inv_mat, source=self.target, target=self.source)
            # 同步缓存，避免再次求逆
            self._inverse_cache._inverse_cache = self
        return self._inverse_cache

    # 矩阵乘法 (组合) ----------------------------------------------------
    def __matmul__(self, other: "Transform") -> "Transform":
        """组合两个变换，返回 new = self @ other。

        按照数学常规：``T = T2 @ T1`` 表示 **先** 执行 ``T1``，再执行 ``T2``，
        等价的组合矩阵为 ``T2.matrix @ T1.matrix``。
        """
        if not isinstance(other, Transform):
            raise TypeError("Transform 只能与 Transform 相乘(@)")

        new_matrix = self.matrix @ other.matrix  # 先 other，再 self

        return Transform(new_matrix, source=other.source, target=self.target)

    # 等价写法：self.compose(other) == other @ self
    def compose(self, other: "Transform") -> "Transform":
        """self.compose(other) 表示先 self，再 other。"""
        return other @ self

    # ------------------------------------------------------------------
    # 应用到点 / 向量
    # ------------------------------------------------------------------
    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        """对点集应用变换，pts 形状 (N,3) 或 (3,)。"""
        pts_h = _homogeneous(pts, w=1.0)  # Nx4
        out = (self.matrix @ pts_h.T).T[:, :3]
        return out

    def apply_vectors(self, vecs: np.ndarray) -> np.ndarray:
        """对向量应用变换（忽略平移）。"""
        vecs_h = _homogeneous(vecs, w=0.0)  # Nx4，最后一位 0
        out = (self.matrix @ vecs_h.T).T[:, :3]
        return out

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """对点集应用变换，pts 形状 (N,3) 或 (3,)。"""
        return self.apply_points(pts)
    
    # ------------------------------------------------------------------
    # 字符串表示
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"Transform(matrix=\n{self.matrix},\n source={self.source}, target={self.target})" 