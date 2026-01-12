# vlmeval/configs/loader.py
from __future__ import annotations
from typing import Dict, Any
from importlib.resources import files
import yaml

PKG = "vlmeval.configs"  # where your YAMLs live

def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _read_yaml(relpath: str) -> Dict[str, Any]:
    data = (files(PKG) / relpath).read_text(encoding="utf-8")
    return yaml.safe_load(data) or {}

def load_config(root_pkg: str = "vlmeval.configs", default_file: str = "default.yaml") -> Dict[str, Any]:
    global PKG
    PKG = root_pkg

    base = _read_yaml(default_file)
    includes = base.pop("include", []) or []

    merged: Dict[str, Any] = {}
    for rel in includes:
        merged = _deep_update(merged, _read_yaml(rel))
    merged = _deep_update(merged, base)

    # optional local overrides (ship it or ignore if missing)
    try:
        merged = _deep_update(merged, _read_yaml("local.yaml"))
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # normalize model namespace if someone provided top-level keys
    if "models" not in merged:
        models = {}
        for k in ("ggpt", "florence2", "shikra"):  
            if k in merged:
                models[k] = merged.pop(k)
        merged["models"] = models
    return merged
