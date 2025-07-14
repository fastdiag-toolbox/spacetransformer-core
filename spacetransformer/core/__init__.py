try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations
    __version__ = "dev"

from .space import Space, get_space_from_nifti, get_space_from_sitk
from .transform import Transform
from .pointset_warpers import calc_transform, warp_point, warp_vector
from .relation_check import find_tight_bbox

__all__ = [
    "Space",
    "Transform",
    "calc_transform",
    "warp_point",
    "warp_vector",
    "find_tight_bbox",
    "get_space_from_nifti",
    "get_space_from_sitk",
] 