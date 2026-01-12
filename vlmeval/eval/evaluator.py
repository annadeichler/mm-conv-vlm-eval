import csv
from pathlib import Path
from typing import Iterable, Dict, Any, Callable

from vlmeval.metrics.iou import iou
from vlmeval.utils.viz import draw_boxes

import os
def eval_models(
    dataset: Iterable[Dict[str,Any]],
    runners: Dict[str, Callable],        # {"ggpt": run_fn, "florence2": run_fn}
    results_dir: str,
    base_name: str,
    iou_thresholds=(0.3, 0.5),
):
    out_root = Path(results_dir); out_root.mkdir(parents=True, exist_ok=True)

    fields = ["exp_id","model","phrase","ref_object_id","utterance",
              "img_color_path","img_mask_path","iou","iou_matched",
              *[f"iou@{t}" for t in iou_thresholds],
              *[f"iou@{t}_matched" for t in iou_thresholds],
              "annotated_image_path","n_bboxes"]

    # Prepare per-model writers and annotation dirs
    writers: Dict[str, csv.DictWriter] = {}
    files: Dict[str, Any] = {}
    annot_dirs: Dict[str, Path] = {}

    try:
        for mname in runners.keys():
            model_dir = out_root / mname
            model_dir.mkdir(parents=True, exist_ok=True)

            csv_path = model_dir / f"{base_name}_{mname}.csv"
            f = csv_path.open("w", newline="")
            files[mname] = f

            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            writers[mname] = wr

            # annotation folder next to the CSV
            img_out_dir = model_dir / f"{base_name}_annot"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            annot_dirs[mname] = img_out_dir

        # Run evaluation
        for item in dataset:
            
            img = item["img_path"]
            phrase = item.get("phrase","")
            utterance = item.get("utterance","")
            meta = {"img_path": img, "utterance": utterance, "phrase": phrase}
            if not os.path.exists(img):
                print(f"Image not found: {img}, skipping.")
                continue
                
            for mname, run_fn in runners.items():
                wr = writers[mname]
                img_out_dir = annot_dirs[mname]
                try:
                    boxes, _ = run_fn(img, phrase, meta)
                    print(boxes)
                    print(f"Model {mname} processed {item['exp_id']}")

                    box = boxes[0] if boxes else None
                    i = iou(item.get("gt_box"), box)
                    m_list = [i > t for t in iou_thresholds]
                    i_list = [i if matched else 0.0 for matched, i in zip(m_list, [i]*len(iou_thresholds))]

                    annot_path = str(img_out_dir / f"{item['exp_id']}__{mname}.png")
                    annot = draw_boxes(
                        img,
                        pred_boxes=([box] if box else []),
                        prompt=f"{phrase}/{utterance}",
                        out_path=annot_path,
                        gt_box=item.get("gt_box"),
                        iou_val=i
                    )

                    row = {
                        "exp_id": item["exp_id"],
                        "model": mname,
                        "phrase": phrase,
                        "ref_object_id": item.get("object_id",""),
                        "utterance": utterance,
                        "img_color_path": img,
                        "img_mask_path": item.get("mask_path",""),
                        "iou": round(i, 6),
                        "iou_matched": any(m_list),
                        "annotated_image_path": annot,
                        "n_bboxes": len(boxes),
                    }
                    for t, val, m in zip(iou_thresholds, i_list, m_list):
                        row[f"iou@{t}"] = round(val, 6)
                        row[f"iou@{t}_matched"] = m

                    wr.writerow(row); files[mname].flush()

                except Exception as e:
                    wr.writerow({
                        "exp_id": item.get("exp_id",""), "model": mname,
                        "phrase": phrase,
                        "ref_object_id": item.get("object_id",""),
                        "utterance": utterance,
                        "img_color_path": img,
                        "img_mask_path": item.get("mask_path",""),
                        "iou": 0.0, "iou_matched": False,
                        **{f"iou@{t}": 0.0 for t in iou_thresholds},
                        **{f"iou@{t}_matched": False for t in iou_thresholds},
                        "annotated_image_path": f"ERROR: {e}",
                        "n_bboxes": 0
                    }); files[mname].flush()
    finally:
        # Close all CSV files
        for f in files.values():
            try: f.close()
            except: pass
