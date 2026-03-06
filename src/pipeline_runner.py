from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from . import dataset_io
from .dataset_io import PageEntry, SourceRef
from .evaluation import NormalizationConfig, evaluate_predictions
from .llm_correction import LLMConfig, correct_text, make_diff
from .ocr_model import OCRConfig, TrOCREngine, crop_to_pil
from .pdf_to_images import convert_pdf_to_images
from .preprocessing import PreprocessConfig, preprocess_image
from .text_detection import TextDetectionConfig, detect, draw_overlay, regions_to_dict


STAGES_ALL = ["pdf_to_images", "preprocess", "detect", "ocr", "llm", "eval"]


def _parse_stages(s: str) -> List[str]:
    s = (s or "all").strip().lower()
    if s in {"all", "*"}:
        return STAGES_ALL
    stages = [p.strip() for p in s.split(",") if p.strip()]
    unknown = [x for x in stages if x not in STAGES_ALL]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Known stages: {STAGES_ALL}")
    return stages


def _rel(run_dir: Path, path: Path) -> str:
    return str(path.relative_to(run_dir)).replace("\\", "/")


def _load_detection_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_ground_truth(cfg: Dict[str, Any], base_dir: Path) -> Optional[Dict[str, str]]:
    inputs = cfg.get("inputs", {}) or {}
    gt_path = inputs.get("ground_truth")
    if not gt_path:
        return None
    p = (base_dir / Path(gt_path)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"ground_truth not found: {p}")
    if p.suffix.lower() in {".json"}:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("ground_truth JSON must be a mapping of page_id -> text")
        return {str(k): str(v) for k, v in d.items()}
    raise ValueError("Unsupported ground_truth format. Use a JSON mapping page_id -> text.")


def run_pipeline(
    *,
    config_path: Path,
    stages: List[str],
    run_id_override: Optional[str] = None,
    force_override: Optional[bool] = None,
    resume_override: Optional[bool] = None,
) -> Path:
    base_dir = Path.cwd()
    cfg = dataset_io.load_yaml(config_path)

    runner_cfg = cfg.get("runner", {}) or {}
    results_dir = base_dir / Path(runner_cfg.get("results_dir", "results"))
    run_id = dataset_io.make_run_id(run_id_override or runner_cfg.get("run_id"))
    run_dir = (results_dir / run_id).resolve()
    dataset_io.ensure_dir(run_dir)

    resume = bool(runner_cfg.get("resume", True)) if resume_override is None else bool(resume_override)
    force = bool(runner_cfg.get("force", False)) if force_override is None else bool(force_override)

    effective_cfg = dict(cfg)
    effective_cfg.setdefault("runner", {})
    effective_cfg["runner"] = {**runner_cfg, "run_id": run_id, "results_dir": str(results_dir)}
    effective_cfg["runner"]["resume"] = resume
    effective_cfg["runner"]["force"] = force

    # Snapshot effective config for reproducibility
    config_snapshot = run_dir / "config.effective.yaml"
    dataset_io.write_yaml(config_snapshot, effective_cfg)

    # Manifest: resume if present and requested
    manifest_path = run_dir / "manifest.json"
    if resume and manifest_path.exists() and not force:
        manifest = dataset_io.load_manifest(run_dir)
    else:
        manifest = dataset_io.new_manifest(run_id, config_relpath=_rel(run_dir, config_snapshot))

    # Load inputs
    inputs_cfg = cfg.get("inputs", {}) or {}
    pdfs = dataset_io.list_pdfs(inputs_cfg, base_dir=base_dir)
    selected_pages = dataset_io.parse_page_selection(inputs_cfg)

    pdf_cfg = cfg.get("pdf_to_images", {}) or {}
    dpi = int(pdf_cfg.get("dpi", 300))
    image_format = str(pdf_cfg.get("image_format", "png"))
    poppler_path = pdf_cfg.get("poppler_path")
    poppler_path = str(poppler_path) if poppler_path else None

    pre_cfg = PreprocessConfig.from_dict(cfg.get("preprocessing", {}) or {})
    det_cfg = TextDetectionConfig.from_dict(cfg.get("text_detection", {}) or {})
    ocr_cfg = OCRConfig.from_dict(cfg.get("ocr", {}) or {})
    llm_cfg = LLMConfig.from_dict(cfg.get("llm", {}) or {})

    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", False))
    norm_cfg = NormalizationConfig.from_dict((eval_cfg.get("normalization", {}) or {}))

    # Stage: PDF to images
    if "pdf_to_images" in stages:
        raw_dir = run_dir / "images" / "raw"
        dataset_io.ensure_dir(raw_dir)
        for pdf in pdfs:
            pages = convert_pdf_to_images(
                pdf,
                raw_dir,
                dpi=dpi,
                image_format=image_format,
                poppler_path=poppler_path,
                selected_pages=selected_pages,
            )
            for p in pages:
                page = dataset_io.get_page(manifest, p.page_id)
                if page is None:
                    page = PageEntry(
                        page_id=p.page_id,
                        source=SourceRef(type="pdf", path=str(p.pdf_path), page_number=p.page_number),
                        artifacts={},
                    )
                    dataset_io.upsert_page(manifest, page)
                dataset_io.register_artifact(manifest, p.page_id, "pdf_to_images", "raw_image", _rel(run_dir, p.image_path))

        dataset_io.save_manifest(run_dir, manifest)

    # Helper to list pages in deterministic order
    pages_sorted = sorted(manifest.pages, key=lambda p: p.page_id)

    # Stage: preprocessing
    if "preprocess" in stages:
        pre_dir = run_dir / "images" / "preprocessed"
        dataset_io.ensure_dir(pre_dir)
        for page in pages_sorted:
            raw_rel = page.artifacts.get("pdf_to_images", {}).get("raw_image")
            if not raw_rel:
                raise RuntimeError(f"Missing raw image for page_id={page.page_id}. Run pdf_to_images stage first.")
            raw_path = run_dir / raw_rel

            out_path = pre_dir / f"{page.page_id}.png"
            if resume and out_path.exists() and not force:
                dataset_io.register_artifact(manifest, page.page_id, "preprocess", "image", _rel(run_dir, out_path))
                continue

            img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {raw_path}")
            pre = preprocess_image(img, pre_cfg)
            cv2.imwrite(str(out_path), pre)
            dataset_io.register_artifact(manifest, page.page_id, "preprocess", "image", _rel(run_dir, out_path))

        dataset_io.save_manifest(run_dir, manifest)

    # Stage: detection + crops + overlays
    if "detect" in stages:
        det_dir = run_dir / "detection"
        overlay_dir = det_dir / "overlays"
        crops_dir = det_dir / "crops"
        dataset_io.ensure_dir(overlay_dir)
        dataset_io.ensure_dir(crops_dir)

        for page in pages_sorted:
            pre_rel = page.artifacts.get("preprocess", {}).get("image")
            if not pre_rel:
                raise RuntimeError(f"Missing preprocessed image for page_id={page.page_id}. Run preprocess stage first.")
            pre_path = run_dir / pre_rel
            img = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {pre_path}")

            json_path = det_dir / f"{page.page_id}.json"
            overlay_path = overlay_dir / f"{page.page_id}.png"
            if resume and json_path.exists() and overlay_path.exists() and not force:
                dataset_io.register_artifact(manifest, page.page_id, "detect", "regions_json", _rel(run_dir, json_path))
                dataset_io.register_artifact(manifest, page.page_id, "detect", "overlay", _rel(run_dir, overlay_path))
                continue

            regions = detect(img, det_cfg)
            dataset_io.write_json(json_path, {"page_id": page.page_id, "regions": regions_to_dict(regions)})
            ov = draw_overlay(img, regions)
            cv2.imwrite(str(overlay_path), ov)

            # Emit crops in reading order for downstream OCR (regions or lines)
            page_crop_dir = crops_dir / page.page_id
            dataset_io.ensure_dir(page_crop_dir)
            crop_paths: List[str] = []

            for r in regions:
                if r.lines:
                    for ln in r.lines:
                        x, y, w, h = ln.bbox.as_tuple()
                        crop = img[y : y + h, x : x + w]
                        pth = page_crop_dir / f"line_{r.order:03d}_{ln.order:03d}.png"
                        cv2.imwrite(str(pth), crop)
                        crop_paths.append(_rel(run_dir, pth))
                else:
                    x, y, w, h = r.bbox.as_tuple()
                    crop = img[y : y + h, x : x + w]
                    pth = page_crop_dir / f"region_{r.order:03d}.png"
                    cv2.imwrite(str(pth), crop)
                    crop_paths.append(_rel(run_dir, pth))

            dataset_io.register_artifact(manifest, page.page_id, "detect", "regions_json", _rel(run_dir, json_path))
            dataset_io.register_artifact(manifest, page.page_id, "detect", "overlay", _rel(run_dir, overlay_path))
            dataset_io.register_artifact(manifest, page.page_id, "detect", "crops_dir", _rel(run_dir, page_crop_dir))

        dataset_io.save_manifest(run_dir, manifest)

    # Stage: OCR
    if "ocr" in stages:
        ocr_dir = run_dir / "ocr"
        items_dir = ocr_dir / "items"
        pages_dir = ocr_dir / "pages"
        dataset_io.ensure_dir(items_dir)
        dataset_io.ensure_dir(pages_dir)

        engine = TrOCREngine(ocr_cfg)

        for page in pages_sorted:
            pre_rel = page.artifacts.get("preprocess", {}).get("image")
            det_rel = page.artifacts.get("detect", {}).get("regions_json")
            if not pre_rel or not det_rel:
                raise RuntimeError(f"Missing prerequisites for OCR on page_id={page.page_id}.")

            pre_path = run_dir / pre_rel
            det_path = run_dir / det_rel
            item_json_path = items_dir / f"{page.page_id}.json"
            page_text_path = pages_dir / f"{page.page_id}.txt"
            if resume and item_json_path.exists() and page_text_path.exists() and not force:
                dataset_io.register_artifact(manifest, page.page_id, "ocr", "items_json", _rel(run_dir, item_json_path))
                dataset_io.register_artifact(manifest, page.page_id, "ocr", "page_text", _rel(run_dir, page_text_path))
                continue

            img = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {pre_path}")

            det_data = _load_detection_json(det_path)
            regions = det_data.get("regions", [])
            ocr_items: List[Dict[str, Any]] = []
            pil_crops = []

            for r in regions:
                rb = r["bbox"]
                if r.get("lines"):
                    for ln in r["lines"]:
                        b = ln["bbox"]
                        bbox = (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
                        pil_crops.append(crop_to_pil(img, bbox))
                        ocr_items.append(
                            {
                                "type": "line",
                                "region_order": int(r["order"]),
                                "line_order": int(ln["order"]),
                                "bbox": bbox,
                            }
                        )
                else:
                    bbox = (int(rb["x"]), int(rb["y"]), int(rb["w"]), int(rb["h"]))
                    pil_crops.append(crop_to_pil(img, bbox))
                    ocr_items.append(
                        {
                            "type": "region",
                            "region_order": int(r["order"]),
                            "bbox": bbox,
                        }
                    )

            results = engine.recognize(pil_crops) if pil_crops else []
            if len(results) != len(ocr_items):
                raise RuntimeError("OCR result count mismatch with crop count.")

            # Attach outputs and reconstruct page text in reading order
            page_lines: List[str] = []
            for item, res in zip(ocr_items, results):
                item["text"] = res.text
                item["confidence"] = res.confidence
                page_lines.append(res.text)

            dataset_io.write_json(item_json_path, {"page_id": page.page_id, "items": ocr_items})
            page_text_path.write_text("\n".join(page_lines).strip() + "\n", encoding="utf-8")

            dataset_io.register_artifact(manifest, page.page_id, "ocr", "items_json", _rel(run_dir, item_json_path))
            dataset_io.register_artifact(manifest, page.page_id, "ocr", "page_text", _rel(run_dir, page_text_path))

        dataset_io.save_manifest(run_dir, manifest)

    # Stage: LLM correction
    if "llm" in stages:
        llm_dir = run_dir / "llm" / "pages"
        dataset_io.ensure_dir(llm_dir)
        for page in pages_sorted:
            raw_rel = page.artifacts.get("ocr", {}).get("page_text")
            if not raw_rel:
                raise RuntimeError(f"Missing OCR page text for page_id={page.page_id}. Run ocr stage first.")
            raw_path = run_dir / raw_rel
            raw_text = raw_path.read_text(encoding="utf-8")

            corr_path = llm_dir / f"{page.page_id}_corrected.txt"
            diff_path = llm_dir / f"{page.page_id}.diff"

            if resume and corr_path.exists() and diff_path.exists() and not force:
                dataset_io.register_artifact(manifest, page.page_id, "llm", "corrected_text", _rel(run_dir, corr_path))
                dataset_io.register_artifact(manifest, page.page_id, "llm", "diff", _rel(run_dir, diff_path))
                continue

            corrected, meta = correct_text(raw_text, llm_cfg, metadata={"page_id": page.page_id})
            corr_path.write_text(corrected, encoding="utf-8")
            diff_path.write_text(make_diff(raw_text, corrected), encoding="utf-8")

            dataset_io.register_artifact(manifest, page.page_id, "llm", "corrected_text", _rel(run_dir, corr_path))
            dataset_io.register_artifact(manifest, page.page_id, "llm", "diff", _rel(run_dir, diff_path))

        dataset_io.save_manifest(run_dir, manifest)

    # Stage: evaluation (optional)
    if "eval" in stages:
        metrics_dir = run_dir / "metrics"
        dataset_io.ensure_dir(metrics_dir)

        # Build predictions maps from artifacts
        preds_raw: Dict[str, str] = {}
        preds_corr: Dict[str, str] = {}
        for page in pages_sorted:
            raw_rel = page.artifacts.get("ocr", {}).get("page_text")
            if raw_rel:
                preds_raw[page.page_id] = (run_dir / raw_rel).read_text(encoding="utf-8")
            corr_rel = page.artifacts.get("llm", {}).get("corrected_text")
            if corr_rel:
                preds_corr[page.page_id] = (run_dir / corr_rel).read_text(encoding="utf-8")

        gt = _load_ground_truth(cfg, base_dir=base_dir) if eval_enabled else None
        report = evaluate_predictions(
            ground_truth=gt,
            predictions_raw=preds_raw,
            predictions_corrected=preds_corr if preds_corr else None,
            normalization=norm_cfg,
        )

        metrics_json = metrics_dir / "metrics.json"
        dataset_io.write_json(metrics_json, report)

        # Human-readable summary
        md_lines = ["## Evaluation Summary", ""]
        if report.get("skipped"):
            md_lines.append(f"- **Status**: skipped ({report.get('reason')})")
        else:
            cov = report.get("coverage", {})
            agg = report.get("aggregate", {}) or {}
            md_lines.append(f"- **Coverage**: {cov.get('evaluated', 0)} / {cov.get('total_processed', 0)} pages")
            md_lines.append(f"- **CER (raw, mean)**: {agg.get('cer_raw_mean')}")
            md_lines.append(f"- **WER (raw, mean)**: {agg.get('wer_raw_mean')}")
            if agg.get("cer_corrected_mean") is not None:
                md_lines.append(f"- **CER (corrected, mean)**: {agg.get('cer_corrected_mean')}")
                md_lines.append(f"- **WER (corrected, mean)**: {agg.get('wer_corrected_mean')}")
        md_lines.append("")

        metrics_md = metrics_dir / "metrics.md"
        metrics_md.write_text("\n".join(md_lines), encoding="utf-8")

        # Run-level artifacts (not page-level)
        dataset_io.save_manifest(run_dir, manifest)

    dataset_io.save_manifest(run_dir, manifest)
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Historical printed OCR pipeline runner")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--stages", default="all", help=f"Comma list of stages or 'all'. Options: {STAGES_ALL}")
    ap.add_argument("--run_id", default=None, help="Override run_id")
    ap.add_argument("--force", action="store_true", help="Recompute artifacts even if present")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume behavior")

    args = ap.parse_args(argv)
    stages = _parse_stages(args.stages)
    run_dir = run_pipeline(
        config_path=Path(args.config),
        stages=stages,
        run_id_override=args.run_id,
        force_override=bool(args.force),
        resume_override=not bool(args.no_resume),
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()

