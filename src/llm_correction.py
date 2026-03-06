from __future__ import annotations

import dataclasses
import difflib
from typing import Any, Dict, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    backend: str = "pass_through"  # pass_through | (future backends)
    prompt_version: str = "v1"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLMConfig":
        return LLMConfig(
            enabled=bool(d.get("enabled", False)),
            backend=str(d.get("backend", "pass_through")),
            prompt_version=str(d.get("prompt_version", "v1")),
        )


def prompt_template(version: str = "v1") -> str:
    if version != "v1":
        raise ValueError(f"Unknown prompt_version: {version}")
    return (
        "You are correcting OCR output from early modern printed Spanish documents.\n"
        "Task: Correct OCR recognition errors ONLY.\n"
        "Constraints:\n"
        "- Preserve historical spelling/orthography; do NOT modernize spelling.\n"
        "- Do NOT translate or summarize.\n"
        "- Preserve line breaks and paragraph breaks.\n"
        "- Output ONLY the corrected text.\n"
    )


def make_diff(raw_text: str, corrected_text: str) -> str:
    raw_lines = raw_text.splitlines(keepends=True)
    corr_lines = corrected_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        raw_lines,
        corr_lines,
        fromfile="ocr_raw",
        tofile="llm_corrected",
        lineterm="",
    )
    return "".join(diff)


def correct_text(
    raw_text: str,
    cfg: LLMConfig,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (corrected_text, correction_metadata).

    This module intentionally defaults to a safe, fully offline backend.
    """
    metadata = dict(metadata or {})

    if not cfg.enabled or cfg.backend == "pass_through":
        out = raw_text
        meta = {
            **metadata,
            "enabled": bool(cfg.enabled),
            "backend": cfg.backend,
            "prompt_version": cfg.prompt_version,
            "skipped": True,
            "reason": "LLM disabled or pass-through backend selected.",
        }
        return out, meta

    raise RuntimeError(
        f"LLM backend '{cfg.backend}' is not implemented in this baseline. "
        "Set llm.enabled=false or llm.backend=pass_through to run offline."
    )

