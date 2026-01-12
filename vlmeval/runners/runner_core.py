# vlmeval/runners/runner_core.py
from __future__ import annotations
from typing import Dict, Tuple, Callable

def get_model_block(cfg: Dict, name: str) -> Dict:
    return (cfg.get("models") or {}).get(name, {}) or {}

def resolve_adapter(name: str) -> Tuple[Callable, Callable]:
    # mono VLMs
    if name == "qwen":
        from vlmeval.models.mono_vlm import qwen_adapter
        return qwen_adapter.configure, qwen_adapter.run
    if name == "shikra":
        from vlmeval.models.mono_vlm import shikra_adapter
        return shikra_adapter.configure, shikra_adapter.run
    if name == "kosmos2":
        from vlmeval.models.mono_vlm import kosmos2_adapter
        return kosmos2_adapter.configure, kosmos2_adapter.run
    if name == "ggpt":
        from vlmeval.models.mono_vlm import ggpt_adapter
        return ggpt_adapter.configure, ggpt_adapter.run
    if name == "florence2":
        from vlmeval.models.mono_vlm import florence2_adapter
        return florence2_adapter.configure, florence2_adapter.run

    # detectors
    if name == "groundingdino":
        from vlmeval.models.detectors import groundingdino as groundingdino_det
        return groundingdino_det.configure, groundingdino_det.run
    if name == "mm_groundingdino":
        from vlmeval.models.detectors import mm_groundingdino as mm_groundingdino_det
        return mm_groundingdino_det.configure, mm_groundingdino_det.run

    # bridges
    if name == "gdino_bridge":
        from vlmeval.models.bridges import gdino_bridge
        return gdino_bridge.configure, gdino_bridge.run

    raise ValueError(f"[runner_core] Unknown adapter name: {name}")

def configure_from_block(name: str, block: Dict) -> Tuple[str, Callable]:
    """
    Configure adapter from a models.<name> block. Returns (name, run_fn).
    - Passes {"model": block, "generation": ..., "detection": ...}
    - Merges legacy 'gen' into 'generation' for compatibility.
    """
    cfg_fn, run_fn = resolve_adapter(name)

    # merge generation + legacy gen
    gen = dict(block.get("generation", {}) or {})
    gen_legacy = block.get("gen", {}) or {}
    if gen_legacy:
        gen = {**gen, **gen_legacy}

    cfg_payload = {
        "model": block,
        "generation": gen,
        "detection": block.get("detection", {}) or {},
    }

    # explicit overrides (defensive): prefer merged 'gen'/'generation'
    max_new_tokens = (gen or {}).get("max_new_tokens")
    cfg_fn(
        config=cfg_payload,
        ckpt=block.get("ckpt"),
        device=block.get("device"),
        max_new_tokens=max_new_tokens,
    )
    return name, run_fn
