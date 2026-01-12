# vlmeval/utils/geometry.py
from typing import Tuple, Optional
import numpy as np
Box = Tuple[float,float,float,float]

def clamp_box(box: Box, W: int, H: int) -> Box:
    x1,y1,x2,y2 = box
    x1,x2 = max(0,min(x1,W-1)), max(0,min(x2,W-1))
    y1,y2 = max(0,min(y1,H-1)), max(0,min(y2,H-1))
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    return (float(x1), float(y1), float(x2), float(y2))

def denorm_box(norm_box: Box, W: int, H: int) -> Box:
    x1,y1,x2,y2 = norm_box
    return clamp_box((x1*W, y1*H, x2*W, y2*H), W, H)

def box_center(box: Box) -> Tuple[int,int]:
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)
