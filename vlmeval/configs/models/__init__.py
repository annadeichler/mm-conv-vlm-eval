# vlmeval/run_model/__init__.py
from . import ggpt
from . import shikra   
from . import qwen
from . import kosmos2
from . import florence2

MODEL_REGISTRY = {
    "ggpt": ggpt,
    "shikra": shikra, 
    "qwen": qwen,
    "kosmos2": kosmos2,
    "florence2": florence2,
}

__all__ = ["MODEL_REGISTRY"]
