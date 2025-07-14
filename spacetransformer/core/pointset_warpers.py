from __future__ import annotations
from typing import Tuple, Union
import numpy as np

try:
    import torch
    _has_torch = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _has_torch = False

from .space import Space
from .transform import Transform


ArrayLike = Union[np.ndarray, list, tuple]


def calc_transform(source: Space, target: Space, backward: bool = False) -> Transform:
    """计算 source.index → target.index 的 Transform。

    backward=False: 正向 (source → target)
    backward=True : 反向 (target → source)
    """
    if backward:
        # target → source
        mat = source.from_world_transform.matrix @ target.to_world_transform.matrix
        return Transform(mat, source=target, target=source)
    else:
        # source → target
        mat = target.from_world_transform.matrix @ source.to_world_transform.matrix
        return Transform(mat, source=source, target=target)


def warp_point(
    point_set: Union["torch.Tensor", np.ndarray, ArrayLike],
    source: Space,
    target: Space,
) -> Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]:
    """将点集从 source.index 映射到 target.index。

    返回 (warp_point_set, isin) 其中 isin 为布尔 mask。"""
    istorch = False
    if _has_torch and isinstance(point_set, torch.Tensor):
        device = point_set.device
        istorch = True
        point_set_np = point_set.cpu().numpy()
    else:
        point_set_np = np.asarray(point_set)

    assert point_set_np.ndim == 2 and point_set_np.shape[1] == 3, "point_set shape 必须为 (N,3)"

    T = calc_transform(source, target)
    warp_pts = T.apply_points(point_set_np)

    isin = np.all((warp_pts >= 0) & (warp_pts <= np.array(target.shape)[None] - 1), axis=1)

    if istorch:
        warp_pts_tensor = torch.from_numpy(warp_pts).to(device=device)
        isin_tensor = torch.from_numpy(isin).to(device=device)
        return warp_pts_tensor, isin_tensor
    else:
        return warp_pts, isin


def warp_vector(
    vector_set: Union["torch.Tensor", np.ndarray, ArrayLike],
    source: Space,
    target: Space,
):
    """变换向量集合，不考虑平移。返回与输入同类型。"""
    istorch = False
    if _has_torch and isinstance(vector_set, torch.Tensor):
        device = vector_set.device
        istorch = True
        vec_np = vector_set.cpu().numpy()
    else:
        vec_np = np.asarray(vector_set)

    assert vec_np.ndim == 2 and vec_np.shape[1] == 3

    dtype = vec_np.dtype
    T = calc_transform(source, target)
    warp_vec = T.apply_vectors(vec_np)

    if istorch:
        return torch.from_numpy(warp_vec).to(device=device, dtype=vector_set.dtype)
    else:
        return warp_vec.astype(dtype) 