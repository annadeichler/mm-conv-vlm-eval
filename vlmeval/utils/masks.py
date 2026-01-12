# vlmeval/utils/masks.py
from typing import Optional, Tuple
import numpy as np
import cv2

def get_gt_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"GT mask not found: {path}")
    return (m > 0).astype("uint8")

def check_containment(x: int, y: int, path: str) -> bool:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return False
    m = (m > 0).astype("uint8")
    h, w = m.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(m[int(y), int(x)])

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def triplet_to_int(color_triplet) -> int:
    """
    Accepts [r,g,b] as float(0..1) or uint8(0..255). Returns unique int id.
    """
    arr = np.array(color_triplet)
    if arr.max() <= 1.0:  # normalized -> uint8
        arr = (arr * 255.0).round().astype(np.uint32)
    else:
        arr = arr.astype(np.uint32)
    return int(arr[0] + 256 * arr[1] + 256 * 256 * arr[2])