from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def make_run_id(user_run_id: Optional[str] = None) -> str:
    if user_run_id:
        return user_run_id
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_page_id(source_path: Path, page_number_1idx: int) -> str:
    stem = source_path.stem
    return f"{stem}_p{page_number_1idx:04d}"


def parse_page_selection(cfg: Dict[str, Any]) -> Optional[Set[int]]:
    pages = cfg.get("pages")
    ranges = cfg.get("page_ranges")
    selected: Set[int] = set()

    if pages is not None:
        for p in pages:
            pi = int(p)
            if pi <= 0:
                raise ValueError("pages must be 1-indexed positive integers")
            selected.add(pi)

    if ranges is not None:
        for r in ranges:
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                raise ValueError("page_ranges entries must be [start, end]")
            start, end = int(r[0]), int(r[1])
            if start <= 0 or end <= 0 or end < start:
                raise ValueError("page_ranges must be 1-indexed with end >= start")
            selected.update(range(start, end + 1))

    return selected or None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclasses.dataclass
class SourceRef:
    type: str  # "pdf" | "image"
    path: str
    page_number: Optional[int] = None  # 1-indexed for PDFs


@dataclasses.dataclass
class PageEntry:
    page_id: str
    source: SourceRef
    artifacts: Dict[str, Dict[str, str]]  # stage -> kind -> relative path


@dataclasses.dataclass
class RunManifest:
    run_id: str
    created_at: str
    config_path: str  # relative path within run dir
    pages: List[PageEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config_path": self.config_path,
            "pages": [
                {
                    "page_id": p.page_id,
                    "source": dataclasses.asdict(p.source),
                    "artifacts": p.artifacts,
                }
                for p in self.pages
            ],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunManifest":
        pages: List[PageEntry] = []
        for p in d.get("pages", []):
            src = p.get("source", {})
            pages.append(
                PageEntry(
                    page_id=p["page_id"],
                    source=SourceRef(
                        type=src["type"],
                        path=src["path"],
                        page_number=src.get("page_number"),
                    ),
                    artifacts=p.get("artifacts", {}) or {},
                )
            )
        return RunManifest(
            run_id=d["run_id"],
            created_at=d["created_at"],
            config_path=d.get("config_path", "config.effective.yaml"),
            pages=pages,
        )


def new_manifest(run_id: str, config_relpath: str = "config.effective.yaml") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=now_utc_iso(),
        config_path=config_relpath,
        pages=[],
    )


def upsert_page(manifest: RunManifest, page: PageEntry) -> None:
    for i, existing in enumerate(manifest.pages):
        if existing.page_id == page.page_id:
            manifest.pages[i] = page
            return
    manifest.pages.append(page)


def get_page(manifest: RunManifest, page_id: str) -> Optional[PageEntry]:
    for p in manifest.pages:
        if p.page_id == page_id:
            return p
    return None


def register_artifact(
    manifest: RunManifest,
    page_id: str,
    stage: str,
    kind: str,
    rel_path: str,
) -> None:
    page = get_page(manifest, page_id)
    if page is None:
        raise KeyError(f"Unknown page_id in manifest: {page_id}")
    page.artifacts.setdefault(stage, {})[kind] = rel_path


def save_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / "manifest.json"
    write_json(path, manifest.to_dict())
    return path


def load_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / "manifest.json"
    d = read_json(path)
    if not isinstance(d, dict):
        raise ValueError("manifest.json must be a JSON object")
    return RunManifest.from_dict(d)


def list_pdfs(inputs_cfg: Dict[str, Any], base_dir: Path) -> List[Path]:
    pdfs: List[Path] = []
    if inputs_cfg.get("pdfs"):
        for p in inputs_cfg["pdfs"]:
            pdfs.append((base_dir / Path(p)).resolve())
    elif inputs_cfg.get("pdf_dir"):
        pdf_dir = (base_dir / Path(inputs_cfg["pdf_dir"])).resolve()
        if not pdf_dir.exists():
            raise FileNotFoundError(f"pdf_dir does not exist: {pdf_dir}")
        pdfs.extend(sorted(pdf_dir.glob("*.pdf")))
    else:
        raise ValueError("inputs must specify either `pdfs` or `pdf_dir`")

    if not pdfs:
        raise FileNotFoundError("No PDFs found in inputs.")
    for p in pdfs:
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
    return pdfs

