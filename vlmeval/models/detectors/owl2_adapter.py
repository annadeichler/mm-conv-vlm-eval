from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading, re, ast, json

import numpy as np
import torch
from PIL import Image

from transformers import OwlViTProcessor, OwlViTForObjectDetection

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "google/owlvit-base-patch32"  # Default use OWL-ViT Base
_dtype: str = "float16"  # auto|float16|bfloat16|float32
_device_map: Optional[str] = None  # auto|None
_device_override: Optional[str] = None  # cuda:0, cuda:1, etc.
_trust_remote_code: bool = True

_model: Optional[OwlViTForObjectDetection] = None
_proc: Optional[OwlViTProcessor] = None
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
    
    print(f"[owl2] configured: ckpt={_ckpt}, device={_device_override}, dtype={_dtype}")

# ------------------- helpers -------------------
def _resolve_ckpt_path() -> str:
    path = _ckpt or os.getenv("OWL2_CKPT") or "google/owlvit-base-patch32"
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

        print(f"[owl2] loading OWL-2 model from: {ckpt}")
        _model = OwlViTForObjectDetection.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = OwlViTProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        if _device_override:
            _model.to(torch.device(_device_override))

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W - 1, x1))); x2 = float(max(0.0, min(W - 1, x2)))
    y1 = float(max(0.0, min(H - 1, y1))); y2 = float(max(0.0, min(H - 1, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

def _build_grounding_prompt(phrase: str, utterance: str = "") -> List[str]:
    """
    Build prompt for grounding task - OWL-2 uses text query list.
    
    We test different referential levels (exact/part/pronominal) by using phrase
    as the target referential expression.
    
    NOTE: OWL-2 is a detector-based model that accepts a list of text queries.
    It performs multi-query object detection by matching each query to image regions.
    We ONLY use phrase and its variants as queries, as extracting keywords from utterance
    would introduce irrelevant queries and interfere with matching.
    
    utterance is provided for reference but not used in the queries.
    """
    phrase = (phrase or "").strip()
    # utterance is not used for OWL-2 (multi-query vocabulary matching)
    
    # OWL-2 uses text query list - use ONLY phrase and its variants
    queries = []
    
    # Primary query: phrase (the target referential expression)
    if phrase:
        queries.append(phrase)
        
        # Add variant of phrase (remove articles/pronouns) to improve detection
        clean_phrase = phrase.replace('the ', '').replace('that ', '').replace('this ', '').replace('a ', '').replace('an ', '')
        if clean_phrase != phrase and clean_phrase:
            queries.append(clean_phrase)
        
        # If plural, add singular form
        if phrase.endswith('s') and len(phrase) > 3:
            singular = phrase[:-1]
            if singular:
                queries.append(singular)
    
    # Deduplicate and limit query count
    unique_queries = list(dict.fromkeys(queries))[:5]  # Maximum 5 queries
    return unique_queries if unique_queries else ["object"]

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    OWL-2 grounding task
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
        text_queries = _build_grounding_prompt(phrase, utterance)
        
        # Prepare input
        inputs = _proc(text=text_queries, images=image, return_tensors="pt")
        
        # Move to correct device
        device = _model.device
        model_dtype = next(_model.parameters()).dtype
        
        # Ensure all inputs are on correct device and unify data types
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in ["input_ids", "attention_mask"]:
                    # These tensors should be integer type
                    inputs[key] = value.to(device, dtype=torch.long)
                else:
                    # All other tensors use model data type
                    inputs[key] = value.to(device, dtype=model_dtype)

        # Perform inference
        outputs = _model(**inputs)
        
        # Post-process results - lower threshold to improve detection rate
        target_sizes = torch.Tensor([[H, W]]).to(device)
        results = _proc.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.01  # Lower confidence threshold
        )[0]
        
        # Extract bounding boxes
        boxes = []
        if len(results['boxes']) > 0:
            # Select highest confidence detection result
            scores = results['scores']
            labels = results['labels']
            
            # Find detection result that best matches target phrase
            best_idx = 0
            best_score = scores[0].item()
            
            for i, (score, label) in enumerate(zip(scores, labels)):
                if score.item() > best_score:
                    best_score = score.item()
                    best_idx = i
            
            # Get best detection result
            bbox = results['boxes'][best_idx]
            detected_label = text_queries[labels[best_idx].item()] if labels[best_idx].item() < len(text_queries) else phrase
            
            print(f"[owl2] detected: '{detected_label}' (score: {best_score:.3f})")
            
            # Convert CUDA tensor to CPU numpy array
            bbox_coords = np.array([bbox[0].cpu().item(), bbox[1].cpu().item(), 
                                   bbox[2].cpu().item(), bbox[3].cpu().item()], dtype=float)
            
            # Ensure coordinates are within image bounds
            bbox_coords = _clip_xyxy(bbox_coords, W, H)
            xyxy: Box = (float(bbox_coords[0]), float(bbox_coords[1]), 
                         float(bbox_coords[2]), float(bbox_coords[3]))
            
            boxes = [xyxy]
        else:
            print(f"[owl2] no objects detected for '{phrase}'")
        
        prompt = f"Text queries: {text_queries}"
        return (boxes, prompt)

    except Exception as e:
        print(f"[owl2] inference error: {e}")
        return ([], prompt if 'prompt' in locals() else "")
