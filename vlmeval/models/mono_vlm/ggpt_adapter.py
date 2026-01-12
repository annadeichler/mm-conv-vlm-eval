# vlmeval/run_model/ggpt.py
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading, re, ast, numpy as np, torch
from PIL import Image
from vlmeval.utils.geometry import denorm_box

Box = Tuple[float,float,float,float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = None
_device_override: Optional[str] = None
_temperature: float = 0.2
_max_new_tokens: int = 512
_cfg: Optional[Dict[str, Any]] = None

def configure(ckpt: Optional[str] = None, device: Optional[str] = None,
              temperature: Optional[float] = None, max_new_tokens: Optional[int] = None,
              config: Optional[Dict[str, Any]] = None) -> None:
    global _ckpt, _device_override, _temperature, _max_new_tokens, _cfg  
    if ckpt: _ckpt = ckpt
    if device: _device_override = device
    if temperature is not None: _temperature = float(temperature)
    if max_new_tokens is not None: _max_new_tokens = int(max_new_tokens)
    if config is not None: _cfg = dict(config)
    reset()

# ------------------- internal state -------------------
_model = None
_tok = None
_img_proc = None
_init_lock = threading.Lock()

# will be filled on demand
IMAGE_TOKEN_INDEX = DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_PATCH_TOKEN = None
DEFAULT_IMAGE_START_TOKEN = DEFAULT_IMAGE_END_TOKEN = None
SeparatorStyle = None
conversation_lib = None
tokenizer_image_token = KeywordsStoppingCriteria = load_image_square = postprocess_output = None
CONFIG = None
load_pretrained_model = None

def _resolve_ckpt_path() -> Path:
    # prefer config; fallback to env; otherwise error
    path = _ckpt or os.getenv("GGPT_CKPT")
    if not path:
        raise RuntimeError(
            "GGPT checkpoint not set. Pass ggpt.configure(ckpt=...) or set GGPT_CKPT."
        )
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"GGPT checkpoint not found: {p}")
    return p

def reset() -> None:
    global _model, _tok, _img_proc
    _model = _tok = _img_proc = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _import_lego() -> None:
    """Import GGPT / lego deps lazily so the module can be imported without them."""
    global IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
    global DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN
    global SeparatorStyle, conversation_lib
    global tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
    global CONFIG, load_pretrained_model

    if load_pretrained_model is not None:
        return  # already imported

    try:
        from lego.constants import (
            IMAGE_TOKEN_INDEX as _ITI, DEFAULT_IMAGE_TOKEN as _DIT,
            DEFAULT_IMAGE_PATCH_TOKEN as _DIPT, DEFAULT_IMAGE_START_TOKEN as _DIST,
            DEFAULT_IMAGE_END_TOKEN as _DIET
        )
        from lego.conversation import SeparatorStyle as _Sep
        from lego import conversation as _conv
        from lego.mm_utils import (
            tokenizer_image_token as _tok_img,
            KeywordsStoppingCriteria as _kw_stop,
            load_image_square as _load_sq,
            postprocess_output as _post
        )
        from lego.model.builder import CONFIG as _CONF, load_pretrained_model as _load
    except Exception as e:
        raise ImportError(
            "GGPT / lego dependencies not available. "
            "Install GroundingGPT (which provides 'lego') or disable the 'ggpt' model."
        ) from e

    IMAGE_TOKEN_INDEX = _ITI
    DEFAULT_IMAGE_TOKEN = _DIT
    DEFAULT_IMAGE_PATCH_TOKEN = _DIPT
    DEFAULT_IMAGE_START_TOKEN = _DIST
    DEFAULT_IMAGE_END_TOKEN = _DIET
    SeparatorStyle = _Sep
    conversation_lib = _conv
    tokenizer_image_token = _tok_img
    KeywordsStoppingCriteria = _kw_stop
    load_image_square = _load_sq
    postprocess_output = _post
    CONFIG = _CONF
    load_pretrained_model = _load

        
def _lazy_load() -> None:
    global _model, _tok, _img_proc
    if _model is not None:
        return
    with _init_lock:
        if _model is not None:
            return
        # _set_videollama_cache_from_env()  
        _import_lego()
        ckpt = _resolve_ckpt_path()
        print(f"[ggpt] loading model from: {ckpt}")
        _model, _tok, _img_proc, _, _ = load_pretrained_model(str(ckpt))
        if _device_override:
            _model.to(_device_override)
            
def _build_prompt(model, utter: str, phrase: str) -> tuple[str, str]:
    q = (f"Given the following utterance: {utter} "
         f"Identify the object referred to by the word {phrase} in the image. "
         f"Return the bounding box.")
    conv = conversation_lib.default_conversation.copy()
    roles = conv.roles
    if model.config.mm_use_im_start_end:
        inp = (DEFAULT_IMAGE_START_TOKEN
               + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len
               + DEFAULT_IMAGE_END_TOKEN + "\n" + q)
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + q
    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop

def _parse_bbox_text(raw: str) -> Optional[np.ndarray]:
    try:
        arr = np.array(ast.literal_eval(raw), dtype=float)
        if arr.shape == (4,): return arr
        if arr.ndim == 2 and arr.size >= 4:
            return arr.flatten()[:4]
    except Exception:
        pass
    m = re.search(r'\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?', raw)
    return None if not m else np.array([float(m.group(i)) for i in range(1,5)], dtype=float)

@torch.inference_mode()
def run(img_path: Image.Image, query: str, meta: Dict[str, Any]) -> List[Box]:
    """Returns a single pixel-space box if present; [] on failure."""
    try:
        _lazy_load()
        device = _model.device
        use_fp16 = (device.type == "cuda")

        utter = (meta.get("utterance", "") or "").replace("coach", "couch")
        phrase = (meta.get("phrase", "") or query)
        sq = load_image_square(img_path, _img_proc)
        img_tensor = _img_proc.preprocess(sq, return_tensors="pt")["pixel_values"]
        img_tensor = (img_tensor.half() if use_fp16 else img_tensor.float()).to(device)

        prompt, stop = _build_prompt(_model, utter, phrase)
        input_ids = tokenizer_image_token(prompt, _tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        stopping = KeywordsStoppingCriteria([stop], _tok, input_ids)

        out_ids = _model.generate(
            input_ids, images=img_tensor, do_sample=True,
            temperature=_temperature, max_new_tokens=_max_new_tokens,
            use_cache=True, stopping_criteria=[stopping],
        )
        raw = _tok.decode(out_ids[0, input_ids.shape[1]:]).strip()
        raw = postprocess_output(raw, meta.get("img_path", ""))
        if raw.endswith(stop): raw = raw[:-len(stop)]

        norm = _parse_bbox_text(raw)
        if norm is None or norm.shape[0] != 4:
            return []
        norm = np.clip(norm, 0.0, 1.0)
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        return [denorm_box(tuple(map(float, norm)), W, H)], prompt
    
    except Exception as e:
        print(f"[ggpt] inference error: {e}, ignore for first run.")
        return []
