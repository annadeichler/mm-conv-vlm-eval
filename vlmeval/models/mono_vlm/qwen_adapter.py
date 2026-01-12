# vlmeval/run_model/qwen25vl.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, re, ast, json, threading

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = "Qwen/Qwen2.5-VL-7B-Instruct"
_device_override: Optional[str] = None
# Force single device by default to avoid CPU↔CUDA mixes
_device_map: Optional[str] = None        # "auto" | None
_dtype: str = "auto"                     # "auto" | "bfloat16" | "float16" | "float32"
_trust_remote_code: bool = False

_temperature: float = 0.0
_max_new_tokens: int = 256
_do_sample: Optional[bool] = None        # if None, infer from temperature

_system_prompt_default: str = "You are a helpful assistant."
_output_coord_space_default: str = "pixel"  # "pixel" | "normalized"

_cfg: Optional[Dict[str, Any]] = None

# ------------------- internal state -------------------
_model: Optional[Qwen2_5_VLForConditionalGeneration] = None
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
              temperature: Optional[float] = None, max_new_tokens: Optional[int] = None,
              do_sample: Optional[bool] = None,
              config: Optional[Dict[str, Any]] = None) -> None:
    """
    Accepts:
      - explicit overrides (ckpt/device/temperature/max_new_tokens/do_sample)
      - config={"model": {...}, "generation": {...}} as provided by your runner
    """
    global _ckpt, _device_override, _device_map, _dtype, _trust_remote_code
    global _temperature, _max_new_tokens, _do_sample
    global _system_prompt_default, _output_coord_space_default, _cfg

    if config is not None:
        # store deep copy-ish
        _cfg = dict(config)

        # model-level
        m = _cfg.get("model", {})
        if "ckpt" in m: _ckpt = m["ckpt"]
        if "device" in m: _device_override = m["device"]
        if "device_map" in m: _device_map = m["device_map"]
        if "dtype" in m: _dtype = str(m["dtype"])
        if "trust_remote_code" in m: _trust_remote_code = bool(m["trust_remote_code"])
        if "system_prompt" in m: _system_prompt_default = str(m["system_prompt"])
        if "output_coord_space" in m: _output_coord_space_default = str(m["output_coord_space"]).lower()

        # generation-level
        g = _cfg.get("generation", {})
        if "temperature" in g: _temperature = float(g["temperature"])
        if "max_new_tokens" in g: _max_new_tokens = int(g["max_new_tokens"])
        if "do_sample" in g: _do_sample = bool(g["do_sample"])

    # explicit arg overrides win over config
    if ckpt: _ckpt = ckpt
    if device: _device_override = device
    if temperature is not None: _temperature = float(temperature)
    if max_new_tokens is not None: _max_new_tokens = int(max_new_tokens)
    if do_sample is not None: _do_sample = bool(do_sample)

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
    path = _ckpt or os.getenv("QWEN_VL_CKPT") or "Qwen/Qwen2.5-VL-7B-Instruct"
    p = Path(path)
    if p.exists():
        return str(p)
    return path  # assume HF id

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

        # Prefer a single explicit device when available to avoid CPU↔CUDA mixups.
        target_device = None
        if _device_override is not None:
            device_map = None
            target_device = torch.device(_device_override)
        elif torch.cuda.is_available():
            device_map = None
            target_device = torch.device("cuda:0")
        else:
            target_device = torch.device("cpu")

        print(f"[qwen25vl] loading model from: {ckpt}")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=_trust_remote_code,
        )
        _proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=_trust_remote_code)

        # If not using accelerate sharding, ensure all weights sit on one device.
        if device_map is None:
            _model.to(target_device)

def _model_device() -> torch.device:
    # robust way to get the actual device hosting the parameters
    try:
        return next(_model.parameters()).device  # type: ignore[arg-type]
    except Exception:
        return getattr(_model, "device", torch.device("cpu"))

_JSON_FENCE_RE = re.compile(r"```json(.*?)```", re.DOTALL)
_NUM4_RE = re.compile(
    r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)

def _build_bbox_prompt(phrase: str, utter: str = "", want_norm: bool = False) -> str:
    phrase = (phrase or "").strip()
    utter  = (utter or "").strip()
    coord_space = "normalized" if want_norm else "pixel"
    hint = f'Utterance: "{utter}"\n' if utter else ""
    return f"""
{hint}Locate the object referred to by "{phrase}" in the image.

Return ONLY a JSON array (no extra text) with one item using this schema:
[
  {{
    "label": "{phrase}",
    "bbox_2d": [x1, y1, x2, y2],
    "coord_space": "{coord_space}"
  }}
]
"""

def _extract_json_block(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def _best_effort_parse_list_of_dicts(text: str) -> Optional[list]:
    blob = _extract_json_block(text)
    try:
        obj = json.loads(blob)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(blob)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    end_idx = blob.rfind('"}')
    if end_idx != -1:
        try:
            obj = ast.literal_eval(blob[:end_idx + 2] + "]")
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    m = _NUM4_RE.search(text)
    if m:
        vals = [float(m.group(i)) for i in range(1, 5)]
        return [{"label": "", "bbox_2d": vals, "coord_space": "normalized"}]
    return None

def _clip_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(max(0.0, min(W, x1))); x2 = float(max(0.0, min(W, x2)))
    y1 = float(max(0.0, min(H, y1))); y2 = float(max(0.0, min(H, y2)))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)

# ------------------- main entry -------------------
@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    Returns (boxes, prompt).
    - 'boxes': list with ONE (x1,y1,x2,y2) in original image pixel coords, or [] on failure.
    - 'prompt': the exact text fed to the model.
    """
    try:
        _lazy_load()

        utter = (meta.get("utterance", "") or "").strip()
        phrase = (meta.get("phrase", "") or query or "").strip()

        # prefer meta override; otherwise the configured default
        system_prompt = meta.get("system_prompt", _system_prompt_default)
        default_norm = _output_coord_space_default.startswith("norm")
        want_norm = bool(meta.get("want_normalized", default_norm))

        prompt = _build_bbox_prompt(phrase=phrase, utter=utter, want_norm=want_norm)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
        ]

        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        text = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _proc(text=[text], images=[pil], padding=True, return_tensors="pt")

        # Move EVERY tensor to the SAME device as model params
        device = _model_device()
        inputs = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        # sampling policy: explicit do_sample wins, else infer from temperature
        do_sample = _do_sample if _do_sample is not None else (_temperature > 0.0)
        temp = max(1e-6, float(_temperature))

        gen_ids = _model.generate(  # type: ignore[arg-type]
            **inputs,
            do_sample=do_sample,
            temperature=temp,
            max_new_tokens=int(_max_new_tokens),
        )

        cut = inputs["input_ids"].shape[-1]
        out = _proc.batch_decode(
            gen_ids[:, cut:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        lst = _best_effort_parse_list_of_dicts(out)
        if not lst:
            return ([], prompt)

        item = lst[0] if isinstance(lst, list) and len(lst) > 0 else None
        if not isinstance(item, dict) or "bbox_2d" not in item:
            return ([], prompt)

        b = np.array(item["bbox_2d"], dtype=float).reshape(-1)[:4]
        coord_space = str(item.get("coord_space", "pixel")).lower().strip()
        if coord_space.startswith("norm"):
            b = np.array([b[0]*W, b[1]*H, b[2]*W, b[3]*H], dtype=float)

        b = _clip_xyxy(b, W, H)
        xyxy: Box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return ([xyxy], prompt)

    except Exception as e:
        print(f"[qwen25vl] inference error: {e}")
        return ([], "")
