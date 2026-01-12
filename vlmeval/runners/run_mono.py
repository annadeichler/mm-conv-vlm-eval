# run.py (mono)
from pathlib import Path
import os
from vlmeval.configs.loader import load_config
from vlmeval.datasets.json_mmconv_referential import JsonReferential
from vlmeval.eval.evaluator import eval_models
from vlmeval.runners.runner_core import configure_from_block, resolve_adapter

def build_runners(cfg):
    runners = {}
    mcfg = cfg.get("models", {}) or {}
    order = (cfg.get("run", {}) or {}).get("models")
    if not order:
        order = [name for name, conf in mcfg.items() if conf.get("enabled", False)]

    for name in order:
        block = mcfg.get(name, {}) or {}

        hf_cache = block.get("hf_cache")
        if hf_cache:
            os.environ.setdefault("HF_HOME", hf_cache)
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)

        # generic path for qwen/shikra/kosmos2/ggpt (and any future ones)
        _, run_fn = configure_from_block(name, block)
        print(f"Configured runner for model '{name}': {run_fn}")
        runners[name] = run_fn

    if not runners:
        raise RuntimeError("No models selected. Set run.models or models.*.enabled: true.")
    return runners

def main():
    cfg = load_config("vlmeval.configs")

    dcfg = (cfg.get("datasets") or {}).get("mmconv_referential", {}) or {}
    json_path = dcfg.get("json", cfg.get("json"))
    root = dcfg.get("root", cfg.get("root"))
    if not json_path or not root:
        raise ValueError("Dataset paths not set. Provide datasets.mmconv_referential.json and .root.")

    ds = JsonReferential(json_path, root)
    runners = build_runners(cfg)
    eval_models(
        dataset=ds,
        runners=runners,
        results_dir=cfg.get("results_dir", "./results"),
        base_name=Path(json_path).stem,
        iou_thresholds=tuple(cfg.get("iou_thresholds", [0.3, 0.5])),
    )

if __name__ == "__main__":
    main()
