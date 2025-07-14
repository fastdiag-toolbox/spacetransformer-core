import json
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Union

import numpy as np
from .transform import Transform


@dataclass
class Space:
    """
    A class representing the geometric space of reference for 3D medical images.

    Stores information about the image's position, orientation, spacing, and dimensions
    in physical space. Can be converted to/from various medical image formats
    (DICOM, NIfTI, SimpleITK).

    Attributes:
        origin: Physical coordinates (x,y,z) of the first voxel in mm
        spacing: Physical size (x,y,z) of each voxel in mm
        x_orientation: Direction cosines of the x-axis
        y_orientation: Direction cosines of the y-axis
        z_orientation: Direction cosines of the z-axis
        shape: Image dimensions (height, width, depth) in voxels
    """

    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float] = field(default_factory=lambda: (0, 0, 0))
    spacing: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 1))
    x_orientation: Tuple[float, float, float] = field(default_factory=lambda: (1, 0, 0))
    y_orientation: Tuple[float, float, float] = field(default_factory=lambda: (0, 1, 0))
    z_orientation: Tuple[float, float, float] = field(default_factory=lambda: (0, 0, 1))

    def __post_init__(self):
        """
        Perform type checking and conversion after initialization.
        Converts numpy arrays to tuples for JSON serialization.
        """
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, np.ndarray):
                # Convert numpy arrays to tuples
                object.__setattr__(self, field_name, tuple(val.tolist()))
        # 初始化缓存变换
        object.__setattr__(self, "_to_world_transform", None)
        object.__setattr__(self, "_from_world_transform", None)

    def to_json(self):
        """
        Serialize the Space object to a JSON string.

        All attributes are already in JSON-serializable types
        (tuple/list/float/int).

        Returns:
            str: JSON string representation of the Space
        """
        return json.dumps(asdict(self))

    def reverse_axis_order(self):
        """
        将空间信息转换为zyx顺序。用于python的索引顺序。
        """
        new_shape = self.shape[::-1]
        new_origin = self.origin
        new_spacing = self.spacing[::-1]
        new_x_orientation = self.z_orientation
        new_y_orientation = self.y_orientation
        new_z_orientation = self.x_orientation
        return Space(
            new_shape,
            new_origin,
            new_spacing,
            new_x_orientation,
            new_y_orientation,
            new_z_orientation,
        )

    @property
    def shape_zyx(self):
        """
        Get the shape in zyx order.
        """
        return self.shape[::-1]

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Space object from a dictionary.

        Args:
            data: Dictionary containing Space attributes
                 Lists will be converted to tuples

        Returns:
            Space: A new Space instance
        """
        # Convert lists to tuples where needed
        converted_data = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in data.items()
        }
        return cls(**converted_data)

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a Space object from a JSON string.

        Args:
            json_str: JSON string containing Space data

        Returns:
            Space: A new Space instance
        """
        obj_dict = json.loads(json_str)
        return cls.from_dict(obj_dict)

    @classmethod
    def from_sitk(cls, simpleitkimage: "SimpleITK.Image"):
        return get_space_from_sitk(simpleitkimage)
    
    @classmethod
    def from_nifty(cls, niftyimage):
        return get_space_from_nifty(niftyimage)

    def to_sitk_direction(self) -> Tuple[float, ...]:
        """
        Convert orientation vectors to SimpleITK direction matrix format.

        Returns:
            tuple: Direction cosines in column-major order
                  (xx,yx,zx,xy,yy,zy,xz,yz,zz)
        """
        x = self.x_orientation
        y = self.y_orientation
        z = self.z_orientation
        return (x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2])

    def to_nifty_affine(self) -> np.ndarray:
        """
        Convert space information to NIfTI affine transformation matrix.

        The affine matrix combines rotation, scaling, and translation:
        - Rotation from orientation vectors
        - Scaling from voxel spacing
        - Translation from origin

        Returns:
            np.ndarray: 4x4 affine transformation matrix
        """
        # Rotation matrix R from orientation cosines
        R = np.array(
            [self.x_orientation, self.y_orientation, self.z_orientation]
        ).T  # Shape (3, 3)

        # Scaling matrix S from spacing
        S = np.diag(self.spacing)

        # Compute affine transformation matrix
        affine = np.eye(4)
        affine[:3, :3] = np.dot(R, S)
        affine[:3, 3] = self.origin

        return affine

    def to_dicom_orientation(self) -> Tuple[float, ...]:
        """
        Convert orientation vectors to DICOM Image Orientation (Patient) format.

        Returns:
            tuple: Row and column direction cosines concatenated
                  (Xx,Xy,Xz,Yx,Yy,Yz)
        """
        return self.x_orientation + self.y_orientation

    def _orientation_matrix(self) -> np.ndarray:
        """返回以列向量形式组织的 3×3 方向余弦矩阵 R。"""
        return np.column_stack((self.x_orientation, self.y_orientation, self.z_orientation))

    # ------------------------------------------------------------------
    # World ↔ Index 变换 ------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def to_world_transform(self) -> Transform:
        """index → world 的 Transform（懒加载）。"""
        if getattr(self, "_to_world_transform") is None:
            RS = self._orientation_matrix() @ np.diag(self.spacing)
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = RS
            mat[:3, 3] = self.origin
            tw = Transform(mat, source=self, target=None)
            object.__setattr__(self, "_to_world_transform", tw)
        return self._to_world_transform  # type: ignore

    @property
    def from_world_transform(self) -> Transform:
        """world → index 的 Transform（懒加载）。"""
        if getattr(self, "_from_world_transform") is None:
            fw = self.to_world_transform.inverse()
            object.__setattr__(self, "_from_world_transform", fw)
        return self._from_world_transform  # type: ignore

    # ------------------------------------------------------------------
    # 常用几何量 --------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def physical_span(self) -> np.ndarray:
        """图像在物理空间总跨度（mm）。"""
        # (shape-1) 逐轴距离向量
        extent = (np.array(self.shape) - 1)
        return self._orientation_matrix() @ (extent * np.array(self.spacing))

    @property
    def end(self) -> np.ndarray:
        """图像体素空间对角点在 world 坐标。"""
        return np.array(self.origin) + self.physical_span

    # ------------------------------------------------------------------
    # apply_* 链式几何操作 --------------------------------------------
    # ------------------------------------------------------------------
    def apply_flip(self, axis: int) -> "Space":
        """沿指定轴翻转。axis ∈ {0,1,2} 对应 x,y,z。"""
        assert axis in (0, 1, 2), "axis 应为 0/1/2"
        # 计算新 origin
        new_origin = list(self.origin)
        new_origin[axis] = self.end[axis]
        # 翻转方向向量
        R = self._orientation_matrix().copy()
        R[:, axis] *= -1
        # 拆分方向向量
        x_o, y_o, z_o = (R[:, 0], R[:, 1], R[:, 2])
        return Space(
            shape=self.shape,
            origin=tuple(new_origin),
            spacing=self.spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    def apply_swap(self, axis1: int, axis2: int) -> "Space":
        """交换两个轴。"""
        assert axis1 in (0, 1, 2) and axis2 in (0, 1, 2) and axis1 != axis2
        order = [0, 1, 2]
        order[axis1], order[axis2] = order[axis2], order[axis1]
        return self.apply_permute(order)

    def apply_permute(self, axis_order: List[int]) -> "Space":
        """按给定顺序重新排列轴。axis_order 应为对 [0,1,2] 的排列。"""
        assert sorted(axis_order) == [0, 1, 2], "axis_order 必须是 [0,1,2] 的排列"
        R = self._orientation_matrix()[:, axis_order]
        new_shape = tuple(np.array(self.shape)[axis_order])
        new_spacing = tuple(np.array(self.spacing)[axis_order])
        x_o, y_o, z_o = (R[:, 0], R[:, 1], R[:, 2])
        return Space(
            shape=new_shape,
            origin=self.origin,
            spacing=new_spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    def apply_bbox(self, bbox: np.ndarray, include_end: bool = False) -> "Space":
        """裁剪 bbox=(3,2) 返回新 Space。bbox[:,0] 起始索引，bbox[:,1] 结束索引(不含)。"""
        bbox = np.asarray(bbox)
        assert bbox.shape == (3, 2), "bbox 需为 3×2 数组"
        assert np.all(bbox[:, 1] > bbox[:, 0]), "bbox 上界需大于下界"
        # shift world origin
        shift = self._orientation_matrix() @ (bbox[:, 0] * np.array(self.spacing))
        new_origin = tuple(np.array(self.origin) + shift)
        new_shape = bbox[:, 1] - bbox[:, 0]
        if include_end:
            new_shape += 1
        return Space(
            shape=tuple(new_shape.tolist()),
            origin=new_origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_shape(self, shape: Tuple[int, int, int]) -> "Space":
        """仅修改 shape，不改变其他属性。"""
        return Space(
            shape=shape,
            origin=self.origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_spacing(self, spacing: Tuple[float, float, float]) -> "Space":
        """修改 spacing，不改变 shape。"""
        return Space(
            shape=self.shape,
            origin=self.origin,
            spacing=spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_float_bbox(self, bbox: np.ndarray, shape: Tuple[int, int, int]) -> "Space":
        """浮点 bbox 裁剪并重新采样到指定 shape。

        参数
        ------
        bbox: ndarray, 形状 (3,2)。bbox[:,0] 起始下标(可为浮点)，bbox[:,1] 结束下标(可为浮点)。
        shape: 目标体素尺寸 (int,int,int)

        返回
        ------
        Space: 新 Space，具有给定 shape, 其物理区域由 bbox 决定。
        """
        # 输入校验 ------------------------------------------------------
        bbox = np.asarray(bbox, dtype=float)
        assert bbox.shape == (3, 2), "bbox 必须是 3×2 数组"
        assert np.all(bbox[:, 1] >= bbox[:, 0]), "bbox[:,1] 必须 ≥ bbox[:,0]"

        new_shape = np.asarray(shape, dtype=int)
        assert new_shape.shape == (3,), "shape 必须长度为 3"
        assert np.all(new_shape > 0), "shape 各维必须 >0"

        R = self._orientation_matrix()  # 3×3 列向量

        # 1) 新 origin ---------------------------------------------------
        shift = R @ (bbox[:, 0] * np.array(self.spacing))  # world 坐标平移
        new_origin = np.array(self.origin) + shift

        # 2) 新 spacing --------------------------------------------------
        physical_span = np.array(self.spacing) * (bbox[:, 1] - bbox[:, 0])
        tmp = new_shape.astype(float) - 1.0
        tmp[tmp == 0] = 1e-3  # 避免除零
        new_spacing = physical_span / tmp

        # 当目标轴长度为 1 时的特殊处理，与旧实现保持一致 -----------
        singular_axis = new_shape == 1
        if np.any(singular_axis):
            new_spacing[singular_axis] = (
                np.array(self.spacing) * np.array(self.shape) / new_shape
            )[singular_axis]
            # 为保证物理中心一致，需要将 origin 移动到 bbox 中心
            shift2 = R @ (np.array(self.spacing) * (bbox[:, 1] - bbox[:, 0]))
            new_origin[singular_axis] = (new_origin + shift2 / 2)[singular_axis]

        return Space(
            shape=tuple(new_shape.tolist()),
            origin=tuple(new_origin.tolist()),
            spacing=tuple(new_spacing.tolist()),
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def apply_zoom(
        self,
        factor: Union[Tuple[float, float, float], List[float], np.ndarray, float],
        *,
        mode: str = "floor",
        align_corners: bool = True,
    ) -> "Space":
        """按给定倍率缩放 shape。

        仅修改 shape (通过四舍五入策略)，spacing 保持不变；origin 与方向不变。
        与旧版 AffineSpace 行为一致（忽略 align_corners 对 origin/spacing 的微调）。
        """
        assert mode in {"floor", "round", "ceil"}, "mode 必须为 floor/round/ceil"
        if np.isscalar(factor):
            factor = (float(factor),) * 3
        factor_arr = np.asarray(factor, dtype=float)
        assert factor_arr.shape == (3,), "factor 必须为长度 3 或标量"

        if mode == "floor":
            new_shape = np.floor(np.array(self.shape) * factor_arr).astype(int)
        elif mode == "round":
            new_shape = np.round(np.array(self.shape) * factor_arr).astype(int)
        else:  # ceil
            new_shape = np.ceil(np.array(self.shape) * factor_arr).astype(int)

        # 至少为 1
        new_shape[new_shape < 1] = 1

        return self.apply_shape(tuple(new_shape.tolist()))

    # ------------------------------------------------------------------
    # 旋转 --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _axis_angle_rotation(self, axis: int, angle_rad: float) -> np.ndarray:
        """生成绕某坐标轴的旋转矩阵 (3×3)。"""
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        if axis == 0:  # X 轴
            rot = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=float,
            )
        elif axis == 1:  # Y 轴
            rot = np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=float,
            )
        elif axis == 2:  # Z 轴
            rot = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=float,
            )
        else:
            raise ValueError("axis 应为 0/1/2")
        return rot

    def apply_rotate(
        self,
        axis: int,
        angle: float,
        *,
        unit: str = "radian",
        center: str = "center",
    ) -> "Space":
        """围绕指定轴旋转空间。

        参数
        ------
        axis: int, 0/1/2 分别对应 x/y/z 轴
        angle: 旋转角度
        unit: "radian" 或 "degree"
        center: "center"(以体素中心为旋转中心) 或 "origin"(以 world origin)
        """
        assert axis in (0, 1, 2), "axis 必须为 0/1/2"
        assert unit in ("radian", "degree"), "unit 只能是 radian/degree"
        assert center in ("center", "origin"), "center 只能是 center/origin"

        angle_rad = float(angle) if unit == "radian" else float(angle) / 180.0 * np.pi

        R_old = self._orientation_matrix()  # 3×3
        rotm = self._axis_angle_rotation(axis, angle_rad)  # 3×3
        R_new = R_old @ rotm  # 列向量右乘

        # 拆列向量
        x_o, y_o, z_o = (R_new[:, 0], R_new[:, 1], R_new[:, 2])

        if center == "center":
            center_world = np.array(self.origin) + self.physical_span / 2.0
            extent = (np.array(self.shape) - 1) * np.array(self.spacing)
            new_span = R_new @ extent
            new_origin = center_world - new_span / 2.0
        else:
            new_origin = np.array(self.origin)

        return Space(
            shape=self.shape,
            origin=tuple(new_origin.tolist()),
            spacing=self.spacing,
            x_orientation=tuple(x_o),
            y_orientation=tuple(y_o),
            z_orientation=tuple(z_o),
        )

    # ------------------------------------------------------------------
    # 辅助方法 ---------------------------------------------------------
    # ------------------------------------------------------------------
    def copy(self) -> "Space":
        """返回一个值完全相等的新 Space。"""
        return Space(
            shape=self.shape,
            origin=self.origin,
            spacing=self.spacing,
            x_orientation=self.x_orientation,
            y_orientation=self.y_orientation,
            z_orientation=self.z_orientation,
        )

    def contain_pointset_ind(self, pointset_ind: np.ndarray) -> np.ndarray:
        """判断给定索引坐标是否位于空间范围内。

        返回 bool ndarray (N,)
        """
        pts = np.asarray(pointset_ind)
        assert pts.ndim == 2 and pts.shape[1] == 3
        return np.all((pts >= 0) & (pts <= np.array(self.shape)[None] - 1), axis=1)

    def contain_pointset_world(self, pointset_world: np.ndarray) -> np.ndarray:
        """判断给定 world 坐标是否位于空间范围内。"""
        pts_idx = self.from_world_transform.apply_points(pointset_world)
        return self.contain_pointset_ind(pts_idx)


def get_space_from_nifty(
    niftyimage: "NiftiImage"
) -> "Space":
    """
    Create a Space object from a NIfTI image.

    Extracts orientation, spacing, and origin information from the
    NIfTI image.

    Args:
        niftyimage: NIfTI image

    Returns:
        Space: A new Space instance

    Raises:
        ValueError: If affine matrix is not 4x4
    """
    affine = niftyimage.affine
    shape  = niftyimage.shape

    if affine.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4.")

    # Extract direction cosines and spacing
    R = affine[:3, :3]  # Extract rotation part
    spacing = np.linalg.norm(R, axis=0)  # Spacing is column norms
    orientation = R / spacing  # Normalize direction cosines

    origin = tuple(affine[:3, 3].tolist())  # Extract origin
    x_orientation = tuple(orientation[:, 0].tolist())
    y_orientation = tuple(orientation[:, 1].tolist())
    z_orientation = tuple(orientation[:, 2].tolist())

    return Space(
        origin=origin,
        spacing=tuple(spacing.tolist()),
        x_orientation=x_orientation,
        y_orientation=y_orientation,
        z_orientation=z_orientation,
        shape=shape,
    )


def get_space_from_sitk(simpleitkimage):
    """
    Create a Space object from a SimpleITK Image.

    Extracts geometric information including:
    - Origin coordinates
    - Voxel spacing
    - Direction cosines
    - Image dimensions

    Args:
        simpleitkimage: SimpleITK Image object

    Returns:
        Space: A new Space instance with geometry matching the SimpleITK image
    """
    origin = simpleitkimage.GetOrigin()
    spacing = simpleitkimage.GetSpacing()
    d = simpleitkimage.GetDirection()  # Flattened in column-major order
    size = simpleitkimage.GetSize()

    # Extract direction vectors from flattened matrix
    x_orientation = [d[0], d[3], d[6]]
    y_orientation = [d[1], d[4], d[7]]
    z_orientation = [d[2], d[5], d[8]]
    shape = size

    return Space(
        origin=origin,
        spacing=spacing,
        x_orientation=x_orientation,
        y_orientation=y_orientation,
        z_orientation=z_orientation,
        shape=shape,
    )

