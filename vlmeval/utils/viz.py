# vlmeval/utils/viz.py
import os
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
Box = Tuple[float,float,float,float]

def draw_boxes(
    img_path: str,
    pred_boxes: List[Box],
    prompt: str,
    out_path: str,
    gt_box: Optional[Box] = None,
    iou_val: Optional[float] = None,
    overlay_mask_path: Optional[str] = None,  # optional RGB seg image to blend
    alpha: float = 0.5
) -> str:
    img = Image.open(img_path).convert("RGB")
    if overlay_mask_path and os.path.exists(overlay_mask_path):
        import cv2, numpy as np
        base = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = cv2.imread(overlay_mask_path)
        if mask is not None:
            blend = cv2.addWeighted(base, 1-alpha, mask, alpha, 0)
            img = Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))

    W,H = img.size
    dr = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    dr.text((10,10), f"Prompt: {prompt}", fill="yellow", font=font)

    for b in pred_boxes:
        x1,y1,x2,y2 = b
        dr.rectangle([x1,y1,x2,y2], outline="red", width=3)

    if gt_box:
        x1,y1,x2,y2 = gt_box
        dr.rectangle([x1,y1,x2,y2], outline="yellow", width=3)
        dr.text((x1, max(0, y1-14)), "GT", fill="yellow", font=font)

    if iou_val is not None:
        dr.text((10, H-24), f"IoU: {iou_val:.3f}", fill="white", font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path
