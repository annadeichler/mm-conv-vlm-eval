# vlmeval/run_model/groundingdino_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading, re, ast, json

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "IDEA-Research/grounding-dino-tiny"  # Default use tiny version, can also use base version
_device: Optional[str] = None
_device_map: Optional[str] = None
_dtype: str = "float32"  # auto|float16|bfloat16|float32
_trust_remote_code: bool = True

# ------------------- internal state -------------------
_model: Optional[AutoModelForZeroShotObjectDetection] = None
_proc: Optional[AutoProcessor] = None
_init_lock = threading.Lock()

# ------------------- configuration -------------------
def configure(config: Dict[str, Any], ckpt: Optional[str] = None, device: Optional[str] = None, **kwargs) -> None:
    global _ckpt, _device, _device_map, _dtype, _trust_remote_code

    model_cfg = config.get("model", {})

    _ckpt = ckpt or model_cfg.get("ckpt") or _ckpt
    _device = device or model_cfg.get("device") or _device
    _device_map = model_cfg.get("device_map") or _device_map
    _dtype = model_cfg.get("dtype") or _dtype
    _trust_remote_code = model_cfg.get("trust_remote_code", _trust_remote_code)

    print(f"[groundingdino] configured: ckpt={_ckpt}, device={_device}, dtype={_dtype}")

# ------------------- helpers -------------------
def _resolve_ckpt_path() -> str:
    path = _ckpt or os.getenv("GROUNDINGDINO_CKPT") or "IDEA-Research/grounding-dino-tiny"
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
        device_override = _device

        # 如果使用 device_map="auto" 进行多GPU，则忽略 device_override
        if device_map == "auto":
            device_override = None
            print(f"[groundingdino] Using device_map='auto' for multi-GPU support, ignoring device override")

        print(f"[groundingdino] loading GroundingDINO model from: {ckpt}")
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        if device_override:
            _model.to(torch.device(device_override))

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W - 1, x1))); x2 = float(max(0.0, min(W - 1, x2)))
    y1 = float(max(0.0, min(H - 1, y1))); y2 = float(max(0.0, min(H - 1, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

def _build_grounding_prompt(phrase: str, utterance: str = "") -> str:
    """
    Build prompt for grounding task - GroundingDINO uses text query for object detection.
    
    GroundingDINO supports multiple queries separated by "." to improve detection rate.
    We use phrase as the primary query and add variants to improve matching.
    
    Best practices for GroundingDINO:
    - Use multiple related terms separated by "."
    - Add singular/plural variants
    - Remove articles (the, a, an) for better matching
    """
    phrase = (phrase or "").strip()
    if not phrase:
        return "object"
    
    # Build query list with phrase and variants
    queries = [phrase]
    
    # Remove articles for better matching
    clean_phrase = phrase.replace('the ', '').replace('that ', '').replace('this ', '').replace('a ', '').replace('an ', '').strip()
    if clean_phrase != phrase and clean_phrase:
        queries.append(clean_phrase)
    
    # Add singular/plural variants
    if phrase.endswith('s') and len(phrase) > 3:
        singular = phrase[:-1]
        if singular:
            queries.append(singular)
    elif len(phrase) > 3:
        # Try plural form (simple heuristic)
        if not phrase.endswith('s'):
            queries.append(phrase + 's')
    
    # Deduplicate while preserving order
    unique_queries = []
    seen = set()
    for q in queries:
        if q and q.lower() not in seen:
            unique_queries.append(q)
            seen.add(q.lower())
    
    # GroundingDINO format: multiple queries separated by "."
    # Format: "query1 . query2 . query3"
    prompt = " . ".join(unique_queries[:5])  # Limit to 5 queries max
    
    return prompt

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    GroundingDINO grounding task - same task as Florence-2 but using different model
    meta: expects keys
      - phrase: str
      - utterance: str
    Returns (boxes, prompt) where boxes are [(x1,y1,x2,y2)] in pixel coords.
    """
    try:

        _lazy_load()

        # Load image
        image = Image.open(img_path).convert("RGB")
        H, W = image.size[1], image.size[0]  # PIL Image: (width, height)
        
        # Build prompt
        phrase = meta.get("phrase", "")
        utterance = meta.get("utterance", "")
        prompt = _build_grounding_prompt(phrase, utterance)
        
        print(f"[groundingdino] phrase: '{phrase}', prompt: '{prompt}'")
        
        # Prepare input
        inputs = _proc(images=image, text=prompt, return_tensors="pt")
        
        # Move to correct device
        device = _model.device
        model_dtype = next(_model.parameters()).dtype
        
        # Ensure all inputs are on correct device and unify data types
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in ["input_ids", "attention_mask", "token_type_ids", "pixel_mask"]:
                    # These tensors should be integer type
                    inputs[key] = value.to(device, dtype=torch.long)
                else:
                    # All other tensors use model data type
                    inputs[key] = value.to(device, dtype=model_dtype)

        # Perform inference
        outputs = _model(**inputs)
        
        # Post-process results - use lower threshold to improve detection rate
        results = _proc.post_process_grounded_object_detection(
            outputs,
            target_sizes=[(H, W)],
            threshold=0.01  # Lower threshold to improve detection rate
        )[0]
        
        # Extract bounding boxes
        boxes = []
        if len(results['boxes']) > 0:
            # Filter and sort detection results
            valid_detections = []
            for i, (score, label, box) in enumerate(zip(results['scores'], results['labels'], results['boxes'])):
                # Only keep non-empty labels with reasonable confidence
                # Lower threshold to 0.05 to improve detection rate
                if label.strip() and score > 0.05:
                    valid_detections.append((score.item(), label, box))
            
            if valid_detections:
                # Sort by confidence
                valid_detections.sort(key=lambda x: x[0], reverse=True)
                
                # Select best detection result
                best_score, best_label, best_bbox = valid_detections[0]
                print(f"[groundingdino] detected: '{best_label}' (score: {best_score:.3f})")
                
                # Convert CUDA tensor to CPU numpy array
                bbox_coords = np.array([best_bbox[0].cpu().item(), best_bbox[1].cpu().item(), 
                                       best_bbox[2].cpu().item(), best_bbox[3].cpu().item()], dtype=float)
                
                # Ensure coordinates are within image bounds
                bbox_coords = _clip_xyxy(bbox_coords, W, H)
                xyxy: Box = (float(bbox_coords[0]), float(bbox_coords[1]), 
                             float(bbox_coords[2]), float(bbox_coords[3]))
                
                boxes = [xyxy]
            else:
                print(f"[groundingdino] no valid detections")
        else:
            print(f"[groundingdino] no objects detected")
        
        return (boxes, prompt)

    except Exception as e:
        print(f"[groundingdino] inference error: {e}")
        return ([], prompt if 'prompt' in locals() else "")
