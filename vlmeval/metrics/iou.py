# vlmeval/metrics/iou.py
from typing import Optional, Tuple
Box = Tuple[float,float,float,float]

def iou(a: Optional[Box], b: Optional[Box]) -> float:
    if a is None or b is None: return 0.0
    x1,y1,x2,y2 = a
    X1,Y1,X2,Y2 = b
    ix1,iy1 = max(x1,X1), max(y1,Y1)
    ix2,iy2 = min(x2,X2), min(y2,Y2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    A = max(0.0, x2-x1) * max(0.0, y2-y1)
    B = max(0.0, X2-X1) * max(0.0, Y2-Y1)
    U = A + B - inter
    return inter / U if U > 0 else 0.0

def iou_mask_box(mask, box: Box) -> float:
    import numpy as np
    x1,y1,x2,y2 = map(int, box)
    if mask is None: return 0.0
    if x2<=x1 or y2<=y1: return 0.0
    box_mask = np.zeros_like(mask, dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 1
    inter = (mask & box_mask).sum()
    union = (mask | box_mask).sum()
    return float(inter/union) if union > 0 else 0.0
