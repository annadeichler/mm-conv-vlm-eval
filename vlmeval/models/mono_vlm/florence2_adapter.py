# vlmeval/run_model/florence_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "microsoft/Florence-2-large"
_device_override: Optional[str] = None
_device_map: Optional[str] = "auto"
_dtype: str = "float16"
_trust_remote_code: bool = True

# ------------------- generation defaults -------------------
_max_new_tokens: int = 256
_num_beams: int = 1
_do_sample: bool = False
_temperature: float = 0.0
_top_p: float = 1.0
_top_k: int = 50
_repetition_penalty: float = 1.0

_cfg: Optional[Dict[str, Any]] = None

# ------------------- internal state -------------------
_model: Optional[AutoModelForCausalLM] = None
_proc: Optional[AutoProcessor] = None
_init_lock = threading.Lock()

# ------------------- tiny cfg helper -------------------
def _cfg_get(path: str, default=None):
    cur = _cfg or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ------------------- public API -------------------
def configure(
    ckpt: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Accepts either explicit overrides or a dict like:
      config={
        "model": {
          "ckpt": "microsoft/Florence-2-large",
          "device": "cuda:0",
          "device_map": "auto",
          "dtype": "float16",
          "trust_remote_code": true
        },
        "generation": { "max_new_tokens": 256 }
      }
    """
    global _ckpt, _device_override, _device_map, _dtype, _trust_remote_code
    global _max_new_tokens, _num_beams, _do_sample, _temperature, _top_p, _top_k, _repetition_penalty

    if config is not None:
        _cfg = dict(config)
        m = _cfg.get("model", {})
        if "ckpt" in m: _ckpt = m["ckpt"]
        if "device" in m: _device_override = m["device"]
        if "device_map" in m: _device_map = m["device_map"]
        if "dtype" in m: _dtype = str(m["dtype"])
        if "trust_remote_code" in m: _trust_remote_code = bool(m["trust_remote_code"])
        g = _cfg.get("generation", {}) or {}
        if "max_new_tokens" in g: _max_new_tokens = int(g["max_new_tokens"])
        if "num_beams" in g: _num_beams = int(g["num_beams"])
        if "do_sample" in g: _do_sample = bool(g["do_sample"])
        if "temperature" in g: _temperature = float(g["temperature"])
        if "top_p" in g: _top_p = float(g["top_p"])
        if "top_k" in g: _top_k = int(g["top_k"])
        if "repetition_penalty" in g: _repetition_penalty = float(g["repetition_penalty"])
    ...
    if ckpt: _ckpt = ckpt
    if device: _device_override = device
    if max_new_tokens is not None: _max_new_tokens = int(max_new_tokens)

    reset()

def reset() -> None:
    """Drop loaded state (for hot-reload during experiments)."""
    global _model, _proc
    _model = None
    _proc = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------- helpers -------------------
def _resolve_ckpt_path() -> str:
    path = _ckpt or os.getenv("FLORENCE2_CKPT") or "microsoft/Florence-2-large"
    p = Path(path)
    return str(p) if p.exists() else path

def _dtype_str_to_torch(dtype_str: str):
    s = (dtype_str or "auto").lower()
    if s == "auto": return "auto"
    if s in ("float16", "fp16"): return torch.float16
    if s in ("bfloat16", "bf16"): return torch.bfloat16
    if s in ("float32", "fp32"): return torch.float32
    return "auto"

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

        print(f"[florence] loading model from: {ckpt}")
        _model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        if _device_override:
            _model.to(torch.device(_device_override))

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W - 1, x1))); x2 = float(max(0.0, min(W - 1, x2)))
    y1 = float(max(0.0, min(H - 1, y1))); y2 = float(max(0.0, min(H - 1, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

def _build_prompt(phrase: str, utter: str) -> str:
    # Florence expects a task token + structured fields.
    phrase = (phrase or "").strip()
    utter = (utter or "").strip()
    return (
        "<CAPTION_TO_PHRASE_GROUNDING>\n"
        f"Caption: {utter}\n"
        f"Phrase: {phrase}"
    )

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    Florence-2 <CAPTION_TO_PHRASE_GROUNDING>
    meta: expects keys
      - phrase: str
      - utterance: str
    Returns (boxes, prompt) where boxes are [(x1,y1,x2,y2)] in pixel coords.
    """
    try:
        _lazy_load()

        phrase = (meta.get("phrase") or query or "").strip()
        utter = (meta.get("utterance") or "").strip()
        if not phrase:
            return ([], "")

        prompt = _build_prompt(phrase, utter)

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        dev = _model.device
        model_dtype = next(_model.parameters()).dtype
        inputs = _proc(text=_build_prompt(phrase, utter), images=image, return_tensors="pt").to(dev, model_dtype)

        gen_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(_max_new_tokens),
            num_beams=int(_num_beams),
            do_sample=bool(_do_sample),
            # temperature=max(1e-6, float(_temperature)) if _do_sample else None,
            # top_p=float(_top_p) if _do_sample else None,
            # top_k=int(_top_k) if _do_sample else None,
            # repetition_penalty=float(_repetition_penalty),
        )
        # Florence provides a helper to parse its own outputs:
        generated_text = _proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
        # Use Florence's official post-processor to get pixel-space bboxes
        parsed = _proc.post_process_generation(
            generated_text,
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(W, H),
        )
        bbs = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>").get("bboxes", [])
        if not bbs:
            print("[florence] no boxes found in output")
            return ([], prompt)
  
        b = np.array(bbs[0], dtype=float).reshape(-1)[:4]
        b = _clip_xyxy(b, W, H)
        xyxy: Box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return ([xyxy], prompt)

    except Exception as e:
        print(f"[florence] inference error: {e}")
        return ([], "")
