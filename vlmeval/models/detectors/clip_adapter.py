from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading, re, ast, json

import numpy as np
import torch
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "openai/clip-vit-base-patch32"  # Default use CLIP ViT-B/32
_dtype: str = "float16"  # auto|float16|bfloat16|float32
_device_map: Optional[str] = None  # auto|None
_device_override: Optional[str] = None  # cuda:0, cuda:1, etc.
_trust_remote_code: bool = True

_model: Optional[CLIPModel] = None
_proc: Optional[CLIPProcessor] = None
_init_lock = threading.Lock()

# ------------------- configuration -------------------
def configure(config: Dict[str, Any], ckpt: Optional[str] = None, **kwargs) -> None:
    global _ckpt, _dtype, _device_map, _device_override, _trust_remote_code
    model_config = config.get("model", {})
    _ckpt = ckpt or model_config.get("ckpt", _ckpt)
    _dtype = model_config.get("dtype", _dtype)
    _device_map = model_config.get("device_map", _device_map)
    _device_override = model_config.get("device", _device_override)
    _trust_remote_code = model_config.get("trust_remote_code", _trust_remote_code)
    
    print(f"[clip] configured: ckpt={_ckpt}, device={_device_override}, dtype={_dtype}")

# ------------------- helpers -------------------
def _resolve_ckpt_path() -> str:
    path = _ckpt or os.getenv("CLIP_CKPT") or "openai/clip-vit-base-patch32"
    p = Path(path)
    return str(p) if p.exists() else path

def _dtype_str_to_torch(dtype_str: str):
    s = (dtype_str or "auto").lower()
    if s == "auto": return "auto"
    if s in ("float16", "fp16"): return torch.float16
    if s in ("bfloat16", "bf16"): return torch.bfloat16
    if s in ("float32", "fp32"): return torch.float32
    return "auto"

# ------------------- functions -------------------
def _lazy_load() -> None:
    global _model, _proc
    if _model is not None and _proc is not None:
        return
    with _init_lock:
        if _model is not None and _proc is not None:
            return

        ckpt = _resolve_ckpt_path()
        dtype = _dtype_str_to_torch(_dtype)
        device_map = _device_map
        if _device_override:
            device_map = None  # explicit device means we place the whole model there

        print(f"[clip] loading CLIP model from: {ckpt}")
        _model = CLIPModel.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = CLIPProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        if _device_override:
            device = torch.device(_device_override)
            # If CUDA_VISIBLE_DEVICES is set, the visible GPU is mapped to cuda:0
            # So we should use cuda:0 instead of the original physical GPU index
            if device.type == "cuda" and "CUDA_VISIBLE_DEVICES" in os.environ:
                # CUDA_VISIBLE_DEVICES is set, use cuda:0 (the only visible GPU)
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
                print(f"[clip] CUDA_VISIBLE_DEVICES is set, using cuda:0 (mapped from physical GPU {_device_override})")
            _model.to(device)

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W - 1, x1))); x2 = float(max(0.0, min(W - 1, x2)))
    y1 = float(max(0.0, min(H - 1, y1))); y2 = float(max(0.0, min(H - 1, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

def _generate_grid_boxes(image_size: Tuple[int, int], grid_size: int = 8) -> List[Box]:
    """Generate grid-based candidate bounding boxes"""
    W, H = image_size
    boxes = []
    
    # Generate different sized grid boxes
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate grid position
            x1 = (i * W) // grid_size
            y1 = (j * H) // grid_size
            x2 = ((i + 1) * W) // grid_size
            y2 = ((j + 1) * H) // grid_size
            
            boxes.append((x1, y1, x2, y2))
    
    return boxes

def _build_grounding_prompt(phrase: str, utterance: str = "") -> str:
    """
    Build prompt for grounding task - CLIP uses text-image similarity matching.
    
    We test different referential levels (exact/part/pronominal) by using phrase
    as the target referential expression.
    
    NOTE: CLIP is a detector-based model that uses vocabulary-based text-image similarity.
    It matches text queries to image regions based on semantic similarity.
    We ONLY use phrase as the query text, as combining with utterance would interfere
    with the similarity matching process.
    
    utterance is provided for reference but not used in the query.
    """
    phrase = (phrase or "").strip()
    # utterance is not used for CLIP (vocabulary-based similarity matching)
    
    # CLIP uses text-image similarity, so we ONLY use phrase as the query
    # Combining with utterance would interfere with similarity matching
    if phrase:
        return phrase
    else:
        return "object"

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    CLIP grounding task using grid-based approach
    meta: expects keys
      - phrase: str
      - utterance: str
    Returns (boxes, prompt) where boxes are [(x1,y1,x2,y2)] in pixel coords.
    """
    try:
        _lazy_load()

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        phrase = meta.get("phrase", "")
        utterance = meta.get("utterance", "")
        prompt = _build_grounding_prompt(phrase, utterance)
        
        # Generate candidate bounding boxes
        candidate_boxes = _generate_grid_boxes((W, H), grid_size=8)
        
        # Calculate similarity between each candidate box and text
        similarities = []
        
        for box in candidate_boxes:
            x1, y1, x2, y2 = box
            # Crop image region
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Calculate CLIP similarity
            inputs = _proc(
                text=[prompt], 
                images=[cropped_image], 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to correct device
            # Get device - if CUDA_VISIBLE_DEVICES is set, use cuda:0, otherwise use _model.device
            if _device_override:
                device = torch.device(_device_override)
                # If CUDA_VISIBLE_DEVICES is set, the visible GPU is mapped to cuda:0
                if device.type == "cuda" and "CUDA_VISIBLE_DEVICES" in os.environ:
                    device = torch.device("cuda:0")
            elif hasattr(_model, 'device'):
                device = _model.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
            
            # Calculate similarity
            outputs = _model(**inputs)
            similarity = torch.cosine_similarity(
                outputs.text_embeds, 
                outputs.image_embeds, 
                dim=-1
            ).item()
            
            similarities.append(similarity)
        
        # Select box with highest similarity
        if similarities:
            best_idx = np.argmax(similarities)
            best_box = candidate_boxes[best_idx]
            best_similarity = similarities[best_idx]
            
            print(f"[clip] detected: '{phrase}' (similarity: {best_similarity:.3f})")
            
            # Ensure coordinates are within image bounds
            bbox_coords = np.array(best_box, dtype=float)
            bbox_coords = _clip_xyxy(bbox_coords, W, H)
            xyxy: Box = (float(bbox_coords[0]), float(bbox_coords[1]), 
                         float(bbox_coords[2]), float(bbox_coords[3]))
            
            return ([xyxy], prompt)
        else:
            print(f"[clip] no valid detections for '{phrase}'")
            return ([], prompt)

    except Exception as e:
        print(f"[clip] inference error: {e}")
        return ([], prompt if 'prompt' in locals() else "")
