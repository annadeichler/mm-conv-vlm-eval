# run.py (or a small helper under vlmeval/config/builders.py)
from pathlib import Path
from vlmeval.config.loader import load_config
from vlmeval.datasets.json_mmconv_referential import JsonReferential
from vlmeval.eval.evaluator import eval_models


# in run.py (or builders.py)
def build_runners(cfg):
    runners = {}
    mcfg = cfg["models"]
    # Preferred: explicit list in config
    order = cfg.get("run", {}).get("models")
    if not order:
        # fallback: any model with enabled: true
        order = [name for name, conf in mcfg.items() if conf.get("enabled", False)]

    for name in order:
        if name == "ggpt":
            # main.py (or wherever you build runners)
            from vlmeval.models import ggpt
            ggpt.configure(ckpt=cfg.ggpt_ckpt)   # e.g. --ggpt-ckpt ./checkpoints/ggpt
            runners["ggpt"] = ggpt.run            # one-time load happens on first call
 
        elif name == "florence2":
            from vlmeval.models.florence2 import run as run_f2
            f = mcfg["florence2"]
            # optional HF cache envs
            import os
            if f.get("hf_cache"):
                os.environ.setdefault("HF_HOME", f["hf_cache"])
                os.environ.setdefault("TRANSFORMERS_CACHE", f["hf_cache"])
            runners["florence2"] = run_f2
        else:
            raise ValueError(f"Unknown model in run.models: {name}")
    return runners


def main():
    cfg = load_config("vlmeval/config")
    ds = JsonReferential(cfg["json"], cfg["root"])  # from dataset yaml (merged to top)
    runners = build_runners(cfg)
    eval_models(
        dataset=ds,
        runners=runners,
        results_dir=cfg["results_dir"],
        base_name=Path(cfg["json"]).stem,
        iou_thresholds=tuple(cfg["iou_thresholds"]),
    )

if __name__ == "__main__":
    main()
