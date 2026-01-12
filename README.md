# mm-conv-vlm-eval — VLM Grounding Evaluation

Framework to **evaluate Vision-Language Models (VLMs)** on **referential expression grounding** (image + text → bounding box).  
Supports **GroundingGPT (GGPT)** and other models through a unified runner API. Logs IoU metrics and saves annotated images.

---
## Models in repo
implemented:GroundingGpt, Shikra,Kosmos-2,Qwen2.5,Florence2
to be implemented: GroundindDino, Blip, Owl2
--

## Features

- **Multi-model evaluation**: run GGPT, Florence-2, Molmo, etc. under one API.  
- **IoU metrics**: configurable thresholds (default `0.3`, `0.5`).  
- **Per-model results**: annotated images saved to `results/<model>/<base_name>_annot/`.  
- **CSV logging**: all models and examples in one `results/<base_name>_eval.csv`.

---

## Installation

```bash
# 1. Create environment
conda create -n mmconv-vlm python=3.10 -y
conda activate mmconv-vlm

# 2. Install in editable mode (from pyproject.toml)
pip install -e .
```

## Run with configs
```bash
python vlmeval/runners/run_mono.py
```
set which model to run in 
```bash
vlmeval/configs/default.yaml
vlmevak/configs/eval.yaml
```
## References to papers in repo
**GroundingGPT: Language-Enhanced Multi-modal Grounding Model**  
Zhaowei Li et al
GitHub: [lzw-lzw/GroundingGPT](https://github.com/lzw-lzw/GroundingGPT) :contentReference[oaicite:0]{index=0}
**Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic**  
Chen et al  
GitHub: [shikras/shikra](https://github.com/shikras/shikra) | arXiv: 2306.15195 :contentReference[oaicite:0]{index=0}


## Contributors

Parts of the detector runner implementations were developed as master's coursework by Maoxuan Sha.