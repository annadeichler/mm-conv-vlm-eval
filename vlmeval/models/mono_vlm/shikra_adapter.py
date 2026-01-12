# vlmeval/run_model/shikra.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os, threading, re, ast, numpy as np, torch
from PIL import Image

Box = Tuple[float, float, float, float]

# ------------------- configurable knobs -------------------
_ckpt: Optional[str] = None
_device_override: Optional[str] = None
_max_new_tokens: int = 256
_do_sample: bool = False

# ------------------- internal state -------------------
_model = None
_tokenizer = None
_preproc = None
_model_args = None
_training_args = None
_init_lock = threading.Lock()

# Shikra imports are lazy
load_pretrained_shikra = None
prepare_interactive = None
PlainBoxFormatter = None
expand2square = None
box_xyxy_expand2square = None

# ------------------- public API -------------------
_cfg: Optional[dict] = None

def configure(ckpt: Optional[str] = None, device: Optional[str] = None,
              max_new_tokens: Optional[int] = None, do_sample: Optional[bool] = None,
              config: Optional[dict] = None) -> None:
    global _ckpt, _device_override, _max_new_tokens, _do_sample, _cfg
    print("[shikra] configure() called")
    print(f"  args: ckpt={ckpt}, device={device}, max_new_tokens={max_new_tokens}, do_sample={do_sample}")
    if _cfg is None:
        _cfg = {"model": {}}
    if config is not None:
        _cfg = dict(config)
        if "model" not in _cfg or not isinstance(_cfg["model"], dict):
            _cfg["model"] = {}

    # overlay explicit args (args win over YAML)
    if ckpt:
        _ckpt = ckpt
        _cfg["model"]["ckpt"] = ckpt
    if device:
        _device_override = device
        _cfg["model"]["device"] = device
    if max_new_tokens is not None:
        _max_new_tokens = int(max_new_tokens)
        _cfg["model"]["max_new_tokens"] = _max_new_tokens
    if do_sample is not None:
        _do_sample = bool(do_sample)
        _cfg["model"]["do_sample"] = _do_sample

    print(f"[shikra] configured with ckpt={_cfg_get('model.ckpt')}, device={_cfg_get('model.device')}")
    reset()
        
def reset() -> None:
    """Drop loaded state (for hot-reload during experiments)."""
    global _model, _tokenizer, _preproc, _model_args, _training_args
    _model = _tokenizer = _preproc = _model_args = _training_args = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------- helpers -------------------
def _cfg_get(path: str, default=None):
    cur = _cfg or {}
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _resolve_ckpt_path() -> Path:
    print("Resolving Shikra checkpoint path...")
    path = _cfg_get("model.ckpt", None) or _ckpt or os.getenv("SHIKRA_CKPT")
    if not path:
        raise RuntimeError("Shikra ckpt not set. Pass configure(config=...) or set SHIKRA_CKPT.")
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Shikra checkpoint not found: {p}")
    return p

def _lazy_imports() -> None:
    global load_pretrained_shikra, prepare_interactive, PlainBoxFormatter
    global expand2square, box_xyxy_expand2square  
    if load_pretrained_shikra is not None:
        return
    try:
        from mmengine import Config  # noqa: F401
        from transformers import BitsAndBytesConfig, GenerationConfig as _GenCfg  # noqa: F401
        from mllm.models.builder.build_shikra import load_pretrained_shikra as _load
        from mllm.dataset.builder import prepare_interactive as _prep
        from mllm.dataset.process_function import PlainBoxFormatter as _PBF
        from mllm.dataset.utils.transform import expand2square as _e2s
        from mllm.dataset.utils.transform import box_xyxy_expand2square as _b2s

        load_pretrained_shikra = _load
        prepare_interactive = _prep
        PlainBoxFormatter = _PBF
        expand2square = _e2s
        box_xyxy_expand2square = _b2s
    except Exception as e:
        raise ImportError(
            "Shikra dependencies are missing. Install the Shikra repo that provides 'mllm'."
        ) from e

def _make_square(pil: Image.Image) -> tuple[Image.Image, int, int, int]:
    """Pad to a centered square (RGB) and return (img, S, pad_left, pad_top)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    S = max(w, h)
    canvas = Image.new("RGB", (S, S), (0, 0, 0))
    pad_left = (S - w) // 2
    pad_top = (S - h) // 2
    canvas.paste(pil, (pad_left, pad_top))
    return canvas, S, pad_left, pad_top

_NUM_RE = re.compile(
    r'\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?'
)

def _parse_bbox_text(raw: str) -> Optional[np.ndarray]:
    """Accepts 'x1,y1,x2,y2' or nested lists; returns np.array([x1,y1,x2,y2])."""
    try:
        arr = np.array(ast.literal_eval(raw), dtype=float)
        if arr.ndim == 1 and arr.shape[0] >= 4:
            return arr[:4]
        if arr.ndim >= 2 and arr.size >= 4:
            return arr.reshape(-1)[:4]
    except Exception:
        pass
    m = _NUM_RE.search(raw)
    return None if not m else np.array([float(m.group(i)) for i in range(1, 5)], dtype=float)

def _is_normalized_box(b: np.ndarray) -> bool:
    return np.all(b >= -1e-6) and np.all(b <= 1.0 + 1e-6)

def _clip_xyxy(b: np.ndarray, max_w: float, max_h: float) -> np.ndarray:
    x1, y1, x2, y2 = b
    x1 = float(max(0.0, min(max_w, x1)))
    x2 = float(max(0.0, min(max_w, x2)))
    y1 = float(max(0.0, min(max_h, y1)))
    y2 = float(max(0.0, min(max_h, y2)))
    return np.array([x1, y1, x2, y2], dtype=float)

def _lazy_load() -> None:
    """Load Shikra once, safely, and publish globals atomically."""
    global _model, _tokenizer, _preproc, _model_args, _training_args
    if _model is not None:
        return
    with _init_lock:
        if _model is not None:
            return

        _lazy_imports()
        from mmengine import Config
        from transformers import GenerationConfig

        ckpt_dir = str(_resolve_ckpt_path())
        print(f"[shikra] loading model from {ckpt_dir} ...")

        # Pull model knobs from YAML with safe defaults
        vision_tower    = _cfg_get("model.vision_tower", "openai/clip-vit-large-patch14")
        image_token_len = int(_cfg_get("model.image_token_len", 256))
        mm_use_im_start = bool(_cfg_get("model.mm_use_im_start_end", True))

        _model_args = Config(dict(
            type='shikra', version='v1',
            cache_dir=None, model_name_or_path=ckpt_dir,
            vision_tower=vision_tower,
            pretrain_mm_mlp_adapter=None,
            mm_vision_select_layer=-2, model_max_length=2048,
            freeze_backbone=False, tune_mm_mlp_adapter=False, freeze_mm_mlp_adapter=False,
            is_multimodal=True, sep_image_conv_front=False,
            image_token_len=image_token_len, mm_use_im_start_end=mm_use_im_start,
            target_processor=dict(boxes=dict(type='PlainBoxFormatter')),
            process_func_args=dict(
                conv=dict(type='ShikraConvProcess'),
                target=dict(type='BoxFormatProcess'),
                text=dict(type='ShikraTextProcess'),
                image=dict(type='ShikraImageProcessor'),
            ),
            conv_args=dict(conv_template='vicuna_v1.1', transforms=dict(type='Expand2square'),
                           tokenize_kwargs=dict(truncation_size=None)),
            gen_kwargs_set_pad_token_id=True,
            gen_kwargs_set_bos_token_id=True,
            gen_kwargs_set_eos_token_id=True,
        ))

        _training_args = Config(dict(
            bf16=False,
            fp16=True,
            device=_cfg_get("model.device", "cuda"),
            fsdp=None
        ))

        model, preproc = load_pretrained_shikra(_model_args, _training_args, **{})

        # Dtypes/devices
        if _device_override:
            device = torch.device(_device_override)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not getattr(model, 'is_quantized', False):
            model.to(dtype=torch.float16 if device.type == 'cuda' else torch.float32, device=device)
        if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            vt = model.model.vision_tower[0]
            if not getattr(vt, 'is_quantized', False):
                vt.to(dtype=torch.float16 if device.type == 'cuda' else torch.float32, device=device)

        tokenizer = preproc['text']

        # Optional: align token embeddings if special tokens were added
        try:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

        # Generation config on the local model
        max_new = int(_cfg_get("model.max_new_tokens", _max_new_tokens))
        do_smpl = bool(_cfg_get("model.do_sample", _do_sample))
        model.generation_config = GenerationConfig(
            do_sample=do_smpl,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        preproc['target'] = {'boxes': PlainBoxFormatter()}

        # Publish to globals (atomic under the lock)
        _model = model
        _tokenizer = tokenizer
        _preproc = preproc

@torch.inference_mode()
def run(img_path: str, query: str, meta: Dict[str, Any]) -> Tuple[List[Box], str]:
    """
    Returns (boxes, prompt). 'boxes' is a list of ONE xyxy box in *original image pixel coords*.
    Compatible with eval_models() which expects (boxes, _).
    """
    try:
        _lazy_load()
        device = _model.device
        # Build prompt
        utter = (meta.get("utterance", "") or "").strip()
        phrase = (meta.get("phrase", "") or query or "").strip()
        prompt = (
            f"Please locate the object referred to by the word '{phrase}' in the image <image> "
            # f"given the utterance: {utter}. Return the coordinates as [x1,y1,x2,y2]."
        )

        # Load image and pad to centered square (to match Shikra's expected transform)
        orig = Image.open(img_path).convert("RGB")
        w, h = orig.size
        S = max(w, h)
        square = expand2square(orig) 
        pad_left = (S - w) // 2
        pad_top  = (S - h) // 2
        # Prepare dialog turn
        ds = prepare_interactive(_model_args, _preproc)
        ds.set_image(square)
        ds.append_message(role=ds.roles[0], message=prompt, boxes=[], boxes_seq=[])
        model_inputs = ds.to_model_input()

        # Move to device / dtype
        for k in ("input_ids", "attention_mask"):
            if k in model_inputs:
                model_inputs[k] = model_inputs[k].to(device)
        if "images" in model_inputs:
            images = model_inputs["images"]
            images = images.to(torch.float16 if device.type == "cuda" else torch.float32).to(device)
            model_inputs["images"] = images

        # Generate
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out_ids = _model.generate(**model_inputs)

        # Decode
        cut = model_inputs["input_ids"].shape[-1]
        raw = _tokenizer.batch_decode(out_ids[:, cut:], skip_special_tokens=True)[0].strip()
        print(f"[shikra] raw output: {raw}")
        # Parse -> xyxy (square coords, either normalized [0,1] or pixels)
        b = _parse_bbox_text(raw)
        if b is None or b.shape[0] < 4:
            return ([], prompt)

        if _is_normalized_box(b):
            # normalized to square side length
            b_sq = np.array([b[0]*S, b[1]*S, b[2]*S, b[3]*S], dtype=float)
        else:
            b_sq = b.astype(float)

        # Map from square coords -> original coords by removing padding
        x1s, y1s, x2s, y2s = b_sq.tolist()
        x1o, y1o = x1s - pad_left, y1s - pad_top
        x2o, y2o = x2s - pad_left, y2s - pad_top

        box_orig = _clip_xyxy(np.array([x1o, y1o, x2o, y2o], dtype=float), w, h)
        # Ensure x1<=x2, y1<=y2
        x1, y1, x2, y2 = box_orig.tolist()
        x1, x2 = (min(x1, x2), max(x1, x2))
        y1, y2 = (min(y1, y2), max(y1, y2))

        return ([(float(x1), float(y1), float(x2), float(y2))], prompt)

    except Exception as e:
        print(f"[shikra] inference error: {e}")
        return ([], "")
