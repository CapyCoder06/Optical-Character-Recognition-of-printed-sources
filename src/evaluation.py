from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Tuple

import jiwer


@dataclasses.dataclass(frozen=True)
class NormalizationConfig:
    strip: bool = True
    collapse_whitespace: bool = True

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormalizationConfig":
        return NormalizationConfig(
            strip=bool(d.get("strip", True)),
            collapse_whitespace=bool(d.get("collapse_whitespace", True)),
        )


def normalize_text(text: str, cfg: NormalizationConfig) -> str:
    out = text
    if cfg.strip:
        out = out.strip()
    if cfg.collapse_whitespace:
        out = " ".join(out.split())
    return out


def evaluate_predictions(
    *,
    ground_truth: Optional[Mapping[str, str]],
    predictions_raw: Mapping[str, str],
    predictions_corrected: Optional[Mapping[str, str]] = None,
    normalization: Optional[NormalizationConfig] = None,
) -> Dict[str, Any]:
    """
    Returns a machine-readable report dict.

    - If ground_truth is None/empty, evaluation is skipped.
    - Metrics are computed only for page_ids present in ground_truth.
    """
    norm = normalization or NormalizationConfig()
    if not ground_truth:
        return {
            "skipped": True,
            "reason": "No ground truth provided.",
            "coverage": {"evaluated": 0, "total_processed": len(predictions_raw)},
            "aggregate": None,
            "per_page": {},
        }

    per_page: Dict[str, Any] = {}
    cer_raws = []
    wer_raws = []
    cer_corrs = []
    wer_corrs = []

    evaluated = 0
    for page_id, gt in ground_truth.items():
        if page_id not in predictions_raw:
            continue
        evaluated += 1
        gt_n = normalize_text(gt, norm)
        raw_n = normalize_text(predictions_raw[page_id], norm)

        cer_raw = float(jiwer.cer(gt_n, raw_n))
        wer_raw = float(jiwer.wer(gt_n, raw_n))
        cer_raws.append(cer_raw)
        wer_raws.append(wer_raw)

        entry: Dict[str, Any] = {"cer_raw": cer_raw, "wer_raw": wer_raw}

        if predictions_corrected is not None and page_id in predictions_corrected:
            corr_n = normalize_text(predictions_corrected[page_id], norm)
            cer_corr = float(jiwer.cer(gt_n, corr_n))
            wer_corr = float(jiwer.wer(gt_n, corr_n))
            cer_corrs.append(cer_corr)
            wer_corrs.append(wer_corr)
            entry.update({"cer_corrected": cer_corr, "wer_corrected": wer_corr})

        per_page[page_id] = entry

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "skipped": False,
        "coverage": {"evaluated": evaluated, "total_processed": len(predictions_raw)},
        "aggregate": {
            "cer_raw_mean": _mean(cer_raws),
            "wer_raw_mean": _mean(wer_raws),
            "cer_corrected_mean": _mean(cer_corrs) if cer_corrs else None,
            "wer_corrected_mean": _mean(wer_corrs) if wer_corrs else None,
        },
        "per_page": per_page,
    }

