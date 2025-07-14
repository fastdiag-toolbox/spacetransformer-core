from __future__ import annotations
from typing import Tuple
import numpy as np

from .space import Space
from .pointset_warpers import warp_point


def _orientation_matrix(space: Space) -> np.ndarray:
    return np.column_stack((space.x_orientation, space.y_orientation, space.z_orientation))


def find_tight_bbox(source: Space, target: Space) -> np.ndarray:
    """计算 source 在 target.index 空间中的最小包围盒 (左闭右开)。"""
    s = np.array(source.shape) - 1
    corners = np.array([
        [0, 0, 0],
        [0, 0, s[2]],
        [0, s[1], 0],
        [s[0], 0, 0],
        [s[0], s[1], 0],
        [s[0], 0, s[2]],
        [0, s[1], s[2]],
        [s[0], s[1], s[2]],
    ])

    warp_corners, _ = warp_point(corners, source, target)
    lefts = np.floor(np.min(warp_corners, axis=0))
    rights = np.ceil(np.max(warp_corners, axis=0)) + 1  # 右开

    lefts = np.minimum(np.maximum(lefts, 0), target.shape)
    rights = np.minimum(np.maximum(rights, 0), target.shape)

    tight_bbox = np.stack([lefts, rights]).T.astype(int)
    return tight_bbox


def _check_same_base(source: Space, target: Space) -> bool:
    return bool(np.all(_orientation_matrix(source) == _orientation_matrix(target)))


def _check_same_spacing(source: Space, target: Space) -> bool:
    return bool(np.all(np.array(source.spacing) == np.array(target.spacing)))


def _check_align_corner(source: Space, target: Space) -> bool:
    # 检查 target 的起点/终点是否落在 source 的整数格点
    R = _orientation_matrix(source)
    # 列向量乘以对应 spacing → RS
    M = R * np.array(source.spacing)[None, :]  # 3×3
    offset_origin = np.linalg.solve(M, np.array(target.origin) - np.array(source.origin))
    offset_end = np.linalg.solve(M, np.array(target.end) - np.array(source.origin))
    align_origin = np.linalg.norm(np.round(offset_origin) - offset_origin) < 1e-4
    align_end = np.linalg.norm(np.round(offset_end) - offset_end) < 1e-4
    return bool(align_origin and align_end)


def _check_isin(source: Space, target: Space) -> bool:
    s = np.array(source.shape) - 1
    eight = np.array([
        [0, 0, 0],
        [s[0], 0, 0],
        [0, s[1], 0],
        [0, 0, s[2]],
        [s[0], 0, s[2]],
        [s[0], s[1], 0],
        [0, s[1], s[2]],
        [s[0], s[1], s[2]],
    ])
    eight_world = source.to_world_transform.apply_points(eight)
    isin = target.contain_pointset_world(eight_world)
    return bool(np.all(isin))


def _check_no_overlap(source: Space, target: Space) -> bool:
    tight = find_tight_bbox(source, target)
    left, right = tight[:, 0], tight[:, 1]
    overlap = np.prod(right - left)
    no1 = bool(overlap == 0)
    tight2 = find_tight_bbox(target, source)
    l2, r2 = tight2[:, 0], tight2[:, 1]
    overlap2 = np.prod(r2 - l2)
    return no1 or bool(overlap2 == 0)


def _check_small_enough(source: Space, target: Space, ratio_thresh: float = 0.2):
    tight = find_tight_bbox(source, target)
    left, right = tight[:, 0], tight[:, 1]
    ratio = np.prod(right - left) / np.prod(target.shape)
    is_small = ratio_thresh > ratio > 0
    return is_small, tight


def _check_valid_flip_permute(source: Space, target: Space):
    flag = False
    flip_dims = [0, 0, 0]
    transpose_order = [0, 1, 2]
    base_mult = _orientation_matrix(source).T @ _orientation_matrix(target)
    identity_ax = np.abs(np.abs(base_mult) - 1) < 1e-4
    pre, post = np.where(identity_ax)
    sortidx = np.argsort(pre)
    pre, post = pre[sortidx], post[sortidx]
    if len(pre) < 3:
        return flag, flip_dims, transpose_order

    if set(pre.tolist()) == {0, 1, 2} and set(post.tolist()) == {0, 1, 2}:
        if not np.all(post == np.sort(post)):
            flag = True
            transpose_order = post.tolist()
    for i in range(3):
        flip_dims[i] = base_mult[pre[i], post[i]] < 0
    flag = np.any(flip_dims) or flag

    if flag:
        check_space = source.copy()
        for i, fl in enumerate(flip_dims):
            if fl:
                check_space = check_space.apply_flip(i)
        check_space = check_space.apply_permute(transpose_order)
        if check_space != target:
            flag = False
    return flag, flip_dims, transpose_order 