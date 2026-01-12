# vlmeval/run_model/kosmos2.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import os, re, threading

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

try:
    import yaml  # optional; only needed if you call configure_from_yaml_file
except Exception:
    yaml = None

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "microsoft/kosmos-2-patch14-224"
_device_override: Optional[str] = None
_device_map: Optional[str] = "auto"     # "auto" | None
_dtype: str = "auto"                    # "auto" | "bfloat16" | "float16" | "float32"
_trust_remote_code: bool = False

_max_new_tokens: int = 64
_prompt_style: str = "preceding"        # "preceding" | "full"
_min_left_context: int = 0              # min chars of left context

_cfg: Optional[Dict[str, Any]] = None

# ------------------- internal state -------------------
_model: Optional[Kosmos2ForConditionalGeneration] = None
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
def configure(ckpt: Optional[str] = None, device: Optional[str] = None,
              max_new_tokens: Optional[int] = None,
              prompt_style: Optional[str] = None, min_left_context: Optional[int] = None,
              config: Optional[Dict[str, Any]] = None) -> None:
    """
    Accepts either:
      - explicit overrides (ckpt/device/max_new_tokens/prompt_style/min_left_context)
      - config={"model": {...}, "generation": {...}}  # typical from your runner's YAML
    """
    global _ckpt, _device_override, _device_map, _dtype, _trust_remote_code
    global _max_new_tokens, _prompt_style, _min_left_context, _cfg

    if config is not None:
        _cfg = dict(config)

        m = _cfg.get("model", {}) or {}
        # model-level
        _ckpt = m.get("ckpt", _ckpt)
        _device_override = m.get("device", _device_override)
        _device_map = m.get("device_map", _device_map)
        _dtype = str(m.get("dtype", _dtype))
        _trust_remote_code = bool(m.get("trust_remote_code", _trust_remote_code))
        _prompt_style = str(m.get("prompt_style", _prompt_style)).lower()
        _min_left_context = int(m.get("min_left_context", _min_left_context))
        # allow max_new_tokens at model-level as a fallback
        if "max_new_tokens" in m:
            _max_new_tokens = int(m["max_new_tokens"])

        g = _cfg.get("generation", {}) or {}
        if "max_new_tokens" in g:
            _max_new_tokens = int(g["max_new_tokens"])

    # explicit arg overrides win
    if ckpt: _ckpt = ckpt
    if device: _device_override = device
    if max_new_tokens is not None: _max_new_tokens = int(max_new_tokens)
    if prompt_style is not None: _prompt_style = str(prompt_style).lower()
    if min_left_context is not None: _min_left_context = int(min_left_context)

    reset()

def configure_from_yaml_dict(cfg: Dict[str, Any], key: str = "kosmos2") -> None:
    """
    Convenience: pass the full YAML dict (already loaded) and the model key under 'models'.
    Example cfg structure:
      models:
        kosmos2:
          ckpt: ...
          device: ...
          generation:
            max_new_tokens: 64
    """
    block = (cfg.get("models", {}) or {}).get(key, {}) if isinstance(cfg, dict) else {}
    configure(config={"model": block, "generation": block.get("generation", {})})

def configure_from_yaml_file(yaml_path: str, key: str = "kosmos2") -> None:
    """
    Convenience: load from a YAML file on disk (requires PyYAML).
    """
    if yaml is None:
        raise RuntimeError("PyYAML not available; install pyyaml or use configure(...).")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    configure_from_yaml_dict(cfg, key=key)

def reset() -> None:
    """Drop loaded state (for hot-reload during experiments)."""
    global _model, _proc
    _model = None
    _proc = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------- helpers -------------------
def _resolve_ckpt_path() -> str:
    path = _ckpt or os.getenv("KOSMOS2_CKPT") or "microsoft/kosmos-2-patch14-224"
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
            device_map = None  # don't use accelerate mapping if device is pinned

        print(f"[kosmos2] loading model from: {ckpt}")
        _model = Kosmos2ForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        if _device_override:
            _model.to(torch.device(_device_override))

def _find_span_ci(text: str, span: str):
    """Case-insensitive find; returns (start, end, original_substring) or (None, None, None)."""
    if not text or not span:
        return None, None, None
    i = text.lower().find(span.lower())
    if i == -1:
        return None, None, None
    j = i + len(span)
    return i, j, text[i:j]

def _wrap_phrase_once(utter: str, phrase: str) -> Optional[str]:
    """Wrap the first (case-insensitive) occurrence of phrase in <p>...</p>."""
    i, j, orig = _find_span_ci(utter, phrase)
    if i is None:
        return None
    return utter[:i] + "<p>" + orig + "</p>" + utter[j:]

def _build_kosmos_prompt(phrase: str, utter: str = "", style: str = "preceding",
                         min_left_context: int = 0) -> str:
    """
    Kosmos-2 phrase-in-context grounding prompt.
      style:
        - "preceding": keep only the left context + <p>{phrase}</p>
        - "full": use the full utterance, wrapping only the target span
    """
    phrase = (phrase or "").strip()
    utter  = (utter or "").strip()

    if not phrase and not utter:
        return f"<grounding> <p>{phrase}</p>"

    i, j, orig = _find_span_ci(utter, phrase)

    if style == "preceding" and i is not None:
        left = utter[:i]
        if min_left_context > 0 and len(left) < min_left_context and j is not None:
            style = "full"
        else:
            return f"<grounding> {left}<p>{orig}</p>"

    wrapped = _wrap_phrase_once(utter, phrase)
    if wrapped is not None:
        return f"<grounding> {wrapped}"

    left_hint = utter[:max(0, min(len(utter), 80))]  # short left context
    return f"<grounding> {left_hint} <p>{phrase}</p>"

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W, x1))); x2 = float(max(0.0, min(W, x2)))
    y1 = float(max(0.0, min(H, y1))); y2 = float(max(0.0, min(H, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

def _select_boxes(entities, target_phrase: str):
    """Pick boxes for the entity whose span matches target_phrase (case-insensitive)."""
    if not entities:
        return None
    target_lc = (target_phrase or "").strip().lower()
    best = None
    for span, _, boxes in entities:
        if (span or "").strip().lower() == target_lc:
            return boxes
        if best is None:
            best = boxes  # fallback: first entity
    return best

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    Returns (boxes, prompt).
    - 'boxes': list with ONE (x1,y1,x2,y2) in original image pixel coords, or [] on failure.
    - 'prompt': the exact text fed to the model.
    meta: expects fields like {"utterance": "...", "phrase": "...", "prompt_style": "preceding|full"}
    """
    try:
        _lazy_load()

        utter  = (meta.get("utterance", "") or "").strip()
        phrase = (meta.get("phrase", "") or query or "").strip()

        style = str(meta.get("prompt_style", _prompt_style)).lower()
        min_left_ctx = int(meta.get("min_left_context", _min_left_context))

        prompt = _build_kosmos_prompt(
            phrase=phrase,
            utter=utter,
            style=style,
            min_left_context=min_left_ctx,
        )
        print(f"[kosmos2] prompt: {prompt}")
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        inputs = _proc(text=prompt, images=pil, return_tensors="pt")
        device = _model.device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        gen_ids = _model.generate(
            **inputs,
            max_new_tokens=int(_max_new_tokens),
        )
        generated_text = _proc.batch_decode(gen_ids, skip_special_tokens=True)[0]

        # entities: [(span, (start,end), [(x1,y1,x2,y2), ...]), ...]
        caption, entities = _proc.post_process_generation(generated_text)

        boxes_norm = _select_boxes(entities, phrase)
        if not boxes_norm:
            return ([], prompt)

        x1, y1, x2, y2 = boxes_norm[0]
        b = np.array([x1 * W, y1 * H, x2 * W, y2 * H], dtype=float)
        b = _clip_xyxy(b, W, H)
        xyxy: Box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return ([xyxy], prompt)

    except Exception as e:
        print(f"[kosmos2] inference error: {e}")
        return ([], "")
