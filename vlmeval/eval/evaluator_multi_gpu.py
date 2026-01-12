import os, csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Callable, Iterable, List, Optional

# import your single-GPU evaluator from the same file or module
# from your_module import eval_models


def _default_devices() -> List[int]:
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n))
    except Exception:
        # Fallback to CPU-only env (not recommended for large VLMs)
        return []

def _assign_models_to_gpus(model_names: List[str], devices: Optional[List[int]]) -> Dict[str, int]:
    if not devices:
        devices = _default_devices()
    if not devices:
        raise RuntimeError("No CUDA devices found. Set `devices` or ensure GPUs are visible.")
    mapping = {}
    for i, m in enumerate(model_names):
        mapping[m] = devices[i % len(devices)]
    return mapping

def _configure_model_device(model_name: str, device_str: str, model_configurers: Optional[Dict[str, Callable]]):
    """
    Best-effort: call a per-model `configure(device=...)` if provided.
    Example:
        from vlmeval.run_model import ggpt, florence2, molmo
        model_configurers = {
            "ggpt": ggpt.configure,
            "florence2": florence2.configure,
            "molmo": molmo.configure,
        }
    """
    if model_configurers and model_name in model_configurers:
        try:
            model_configurers[model_name](device=device_str)
        except TypeError:
            # if signature is different, try generic
            try:
                model_configurers[model_name](device_str)
            except Exception:
                pass
        except Exception:
            pass

def _eval_one_model_proc(
    model_name: str,
    run_fn: Callable,
    dataset: Iterable[Dict[str, Any]],
    results_dir: str,
    base_name: str,
    iou_thresholds: tuple,
    device_id: int,
    model_configurers: Optional[Dict[str, Callable]],
):
    # Pin this process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device_str = f"cuda:0"   # inside this proc, visible device index 0 maps to the pinned GPU

    # Let the model/runners know their device (if they expose a configure)
    _configure_model_device(model_name, device_str, model_configurers)

    # Run the single-GPU eval for JUST THIS MODEL
    runners = {model_name: run_fn}
    eval_models(
        dataset=dataset,
        runners=runners,
        results_dir=results_dir,   # eval_models already writes under results/<model>/
        base_name=base_name,
        iou_thresholds=iou_thresholds,
    )

def multi_gpu_eval_per_model(
    dataset: Iterable[Dict[str, Any]],
    runners: Dict[str, Callable],                 # {"ggpt": run_fn, "florence2": run_fn, ...}
    results_dir: str,
    base_name: str,
    iou_thresholds=(0.3, 0.5),
    devices: Optional[List[int]] = None,          # e.g., [0,1,2,3]
    model_configurers: Optional[Dict[str, Callable]] = None,  # per-model configure() fns
    merge_csv: bool = True,
) -> Optional[str]:
    """
    Launches one process per model, each bound to its own GPU.
    Returns path to merged CSV if merge_csv=True, else None.
    """
    results_dir = str(Path(results_dir))
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    model_names = list(runners.keys())
    placement = _assign_models_to_gpus(model_names, devices)

    # Spawn processes (one per model)
    procs: List[mp.Process] = []
    for m in model_names:
        p = mp.Process(
            target=_eval_one_model_proc,
            args=(
                m,
                runners[m],
                list(dataset),       # make sure it's materialized; avoid generator reuse across procs
                results_dir,
                base_name,
                iou_thresholds,
                placement[m],
                model_configurers,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Join all
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker for model exited with code {p.exitcode}")

    # Optional: merge per-model CSVs into one combined file
    if merge_csv:
        merged_path = Path(results_dir) / f"{base_name}_eval.csv"
        _merge_model_csvs(results_dir, base_name, model_names, merged_path)
        return str(merged_path)
    return None

def _merge_model_csvs(results_dir: str, base_name: str, model_names: List[str], out_path: Path):
    """
    Collects CSVs from:
        results/<model>/<base_name>_<model>.csv
    and concatenates into:
        results/<base_name>_eval.csv
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    with out_path.open("w", newline="") as fout:
        writer = None
        for m in model_names:
            csv_path = Path(results_dir) / m / f"{base_name}_{m}.csv"
            if not csv_path.exists():
                # If a model errored early, its CSV might be missing. Skip gracefully.
                continue
            with csv_path.open("r", newline="") as fin:
                reader = csv.DictReader(fin)
                if not header_written:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
