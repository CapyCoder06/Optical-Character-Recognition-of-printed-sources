from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np


ReadingOrder = Literal["single_column", "two_column"]


@dataclasses.dataclass(frozen=True)
class TextDetectionConfig:
    reading_order: ReadingOrder = "single_column"
    min_region_area: int = 5000
    line_segmentation_enabled: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TextDetectionConfig":
        lo = d.get("line_segmentation", {}) or {}
        return TextDetectionConfig(
            reading_order=str(d.get("reading_order", "single_column")),
            min_region_area=int(d.get("min_region_area", 5000)),
            line_segmentation_enabled=bool(lo.get("enabled", False)),
        )


@dataclasses.dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return int(self.w) * int(self.h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))


@dataclasses.dataclass(frozen=True)
class DetectedLine:
    bbox: BBox
    order: int


@dataclasses.dataclass(frozen=True)
class DetectedRegion:
    bbox: BBox
    order: int
    lines: List[DetectedLine]


def _ensure_binary_u8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    # If already close to binary, keep; else Otsu
    unique = np.unique(gray)
    if len(unique) <= 4:
        return gray
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def detect_text_regions(preprocessed: np.ndarray, cfg: TextDetectionConfig) -> List[BBox]:
    """
    Baseline heuristic detector for primary printed text regions.
    Returns a list of bounding boxes in page coordinates.
    """
    binary = _ensure_binary_u8(preprocessed)

    # Invert so text is foreground (white) for morphology.
    fg = 255 - binary

    h, w = fg.shape[:2]
    # Kernel sizes tuned for printed text blocks; configurable later if needed.
    kx = max(15, w // 60)
    ky = max(5, h // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))

    connected = cv2.dilate(fg, kernel, iterations=2)
    connected = cv2.erode(connected, kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BBox] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = BBox(int(x), int(y), int(bw), int(bh))
        if b.area() < cfg.min_region_area:
            continue
        # Ignore extremely thin boxes (likely noise)
        if b.w < 30 or b.h < 30:
            continue
        boxes.append(b)

    return boxes


def order_regions(boxes: List[BBox], cfg: TextDetectionConfig, page_width: int) -> List[BBox]:
    if cfg.reading_order == "two_column":
        mid = page_width / 2.0
        left = [b for b in boxes if (b.x + b.w / 2.0) < mid]
        right = [b for b in boxes if (b.x + b.w / 2.0) >= mid]
        left_sorted = sorted(left, key=lambda b: (b.y, b.x))
        right_sorted = sorted(right, key=lambda b: (b.y, b.x))
        return left_sorted + right_sorted

    # default single column
    return sorted(boxes, key=lambda b: (b.y, b.x))


def segment_lines(preprocessed: np.ndarray, region: BBox) -> List[BBox]:
    """
    Simple line segmentation using horizontal projection profile.
    Returns line boxes in *page* coordinates.
    """
    binary = _ensure_binary_u8(preprocessed)
    x, y, w, h = region.as_tuple()
    crop = binary[y : y + h, x : x + w]
    inv = 255 - crop  # text pixels high

    # Horizontal projection: sum foreground per row
    proj = (inv > 0).sum(axis=1).astype(np.int32)
    if proj.size == 0:
        return []

    # Smooth projection to reduce small gaps
    proj_smooth = cv2.blur(proj.reshape(-1, 1).astype(np.float32), (1, 9)).reshape(-1)
    thresh = max(5.0, float(np.percentile(proj_smooth, 60)))
    is_text = proj_smooth >= thresh

    lines: List[BBox] = []
    in_run = False
    start = 0
    for i, v in enumerate(is_text):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            end = i
            in_run = False
            if end - start >= 10:
                lines.append(BBox(x=x, y=y + start, w=w, h=end - start))
    if in_run:
        end = len(is_text)
        if end - start >= 10:
            lines.append(BBox(x=x, y=y + start, w=w, h=end - start))

    # Merge adjacent lines separated by tiny gaps
    merged: List[BBox] = []
    for b in sorted(lines, key=lambda b: b.y):
        if not merged:
            merged.append(b)
            continue
        prev = merged[-1]
        gap = b.y - (prev.y + prev.h)
        if gap <= 3:
            new_y = prev.y
            new_h = (b.y + b.h) - prev.y
            merged[-1] = BBox(x=prev.x, y=new_y, w=prev.w, h=new_h)
        else:
            merged.append(b)

    return merged


def detect(preprocessed: np.ndarray, cfg: TextDetectionConfig) -> List[DetectedRegion]:
    h, w = preprocessed.shape[:2]
    regions = detect_text_regions(preprocessed, cfg)
    regions = order_regions(regions, cfg, page_width=w)

    detected: List[DetectedRegion] = []
    for idx, r in enumerate(regions):
        line_boxes: List[BBox] = []
        if cfg.line_segmentation_enabled:
            line_boxes = segment_lines(preprocessed, r)
            line_boxes = sorted(line_boxes, key=lambda b: b.y)
        lines = [DetectedLine(bbox=b, order=i) for i, b in enumerate(line_boxes)]
        detected.append(DetectedRegion(bbox=r, order=idx, lines=lines))
    return detected


def draw_overlay(page_gray: np.ndarray, regions: List[DetectedRegion]) -> np.ndarray:
    """
    Returns an RGB overlay image for debugging.
    """
    if page_gray.ndim != 2:
        raise ValueError("Expected grayscale page image")
    overlay = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)

    for r in regions:
        x, y, w, h = r.bbox.as_tuple()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for ln in r.lines:
            lx, ly, lw, lh = ln.bbox.as_tuple()
            cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 1)

    return overlay


def regions_to_dict(regions: List[DetectedRegion]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in regions:
        out.append(
            {
                "order": r.order,
                "bbox": dataclasses.asdict(r.bbox),
                "lines": [{"order": ln.order, "bbox": dataclasses.asdict(ln.bbox)} for ln in r.lines],
            }
        )
    return out

