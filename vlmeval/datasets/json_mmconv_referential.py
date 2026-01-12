from pathlib import Path
from typing import Dict, Iterator
from PIL import Image
import json
import numpy as np
import cv2
import os
from vlmeval.utils.masks import mask_to_bbox

def _load_mask_bbox(mask_path: str):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    return mask_to_bbox((m > 0).astype("uint8"))

class JsonReferential:
    """
    JSON format:
      { exp_id: {
          "object_id": "...",
          "phrase": "...",
          "utterance": "...",
          "image_paths": {"color": "...", "mask": "..."},
          "recording_id": "...", ... (optional extras)
        }, ... }
    """
    def __init__(self, json_path: str, root: str):
        self.json_path = Path(json_path)
        self.root = Path(root)
        self.items: Dict = json.loads(self.json_path.read_text())

    def __len__(self): return len(self.items)

    def __iter__(self) -> Iterator[Dict]:
        for exp_id, data in self.items.items():
            img_path = (self.root / data["image_paths"]["color"]).as_posix()
            mask_path = (self.root / data["image_paths"]["mask"]).as_posix()
            if not Path(img_path).exists() or not Path(mask_path).exists():
                print(f"Image or mask not found for exp_id {exp_id}, skipping.")
                continue
            gt_box = _load_mask_bbox(mask_path)
            yield {
                "exp_id": exp_id,
                "image": Image.open(img_path).convert("RGB"),
                "img_path": img_path,
                "mask_path": mask_path,
                "gt_box": gt_box,
                "utterance": data.get("utterance", "").replace("coach", "couch"),
                "phrase": data.get("phrase", ""),
                "object_id": data.get("object_id", ""),
                "recording_id": data.get("recording_id", None),
            }
