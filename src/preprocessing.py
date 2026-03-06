from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    grayscale: bool = True
    denoise_enabled: bool = True
    denoise_method: str = "median"  # "median" | "bilateral"
    denoise_ksize: int = 3
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    binarize_enabled: bool = True
    binarize_method: str = "adaptive"  # "adaptive" | "otsu"
    deskew_enabled: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PreprocessConfig":
        denoise = d.get("denoise", {}) or {}
        clahe = d.get("clahe", {}) or {}
        binarize = d.get("binarize", {}) or {}
        deskew = d.get("deskew", {}) or {}

        tgs = clahe.get("tile_grid_size", [8, 8])
        if isinstance(tgs, (list, tuple)) and len(tgs) == 2:
            tile = (int(tgs[0]), int(tgs[1]))
        else:
            tile = (8, 8)

        return PreprocessConfig(
            grayscale=bool(d.get("grayscale", True)),
            denoise_enabled=bool(denoise.get("enabled", True)),
            denoise_method=str(denoise.get("method", "median")),
            denoise_ksize=int(denoise.get("ksize", 3)),
            clahe_enabled=bool(clahe.get("enabled", True)),
            clahe_clip_limit=float(clahe.get("clip_limit", 2.0)),
            clahe_tile_grid_size=tile,
            binarize_enabled=bool(binarize.get("enabled", True)),
            binarize_method=str(binarize.get("method", "adaptive")),
            deskew_enabled=bool(deskew.get("enabled", False)),
        )


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.denoise_enabled:
        return gray
    k = max(1, int(cfg.denoise_ksize))
    if k % 2 == 0:
        k += 1

    method = cfg.denoise_method.lower()
    if method == "median":
        return cv2.medianBlur(gray, k)
    if method == "bilateral":
        # Deterministic bilateral filtering with fixed parameters
        return cv2.bilateralFilter(gray, d=k, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"Unknown denoise method: {cfg.denoise_method}")


def apply_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.clahe_enabled:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip_limit),
        tileGridSize=tuple(cfg.clahe_tile_grid_size),
    )
    return clahe.apply(gray)


def binarize(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.binarize_enabled:
        return gray

    method = cfg.binarize_method.lower()
    if method == "adaptive":
        # Adaptive threshold is robust to uneven illumination
        return cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=35,
            C=11,
        )
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    raise ValueError(f"Unknown binarize method: {cfg.binarize_method}")


def _estimate_skew_angle_deg(binary: np.ndarray) -> Optional[float]:
    # Expect text as dark on light; invert to make text as foreground.
    if binary.ndim != 2:
        return None
    inv = 255 - binary
    ys, xs = np.where(inv > 0)
    if len(xs) < 500:
        return None
    coords = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])
    # OpenCV returns angle in [-90, 0); convert to deskew angle.
    if angle < -45:
        angle = 90 + angle
    return angle


def deskew(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if not cfg.deskew_enabled:
        return image

    # Use binarized image to estimate skew, but rotate the current image.
    binary = image
    if binary.ndim != 2:
        binary = to_grayscale(binary)
    # If not already binary-ish, threshold for skew estimation.
    _, thr = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    angle = _estimate_skew_angle_deg(thr)
    if angle is None or abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Returns a preprocessed image in a consistent uint8 grayscale format.
    """
    out = img
    if cfg.grayscale:
        out = to_grayscale(out)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    out = denoise(out, cfg)
    out = apply_clahe(out, cfg)
    out = binarize(out, cfg)
    out = deskew(out, cfg)

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim != 2:
        out = to_grayscale(out)
    return out

