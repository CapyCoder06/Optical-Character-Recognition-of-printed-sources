## Why

Early modern printed Spanish documents have degraded scans, non-standard typography, irregular layouts, and spelling variation that cause high OCR error rates. This change proposes a modular, reproducible end-to-end OCR pipeline that pairs a strong baseline OCR model with LLM-based post-correction to reach and measure ≥ 90% transcription accuracy.

## What Changes

- Add a reproducible pipeline that converts PDFs to images, preprocesses pages, detects main text regions/lines, runs OCR, applies LLM post-correction, and evaluates results with CER/WER.
- Introduce a consistent dataset/label I/O layer so experiments do not hardcode paths and can be rerun on new sources.
- Provide a single CLI entrypoint to run the full pipeline or individual stages, writing artifacts to `results/` for inspection and comparison.

## Capabilities

### New Capabilities

- `dataset-io`: Standardized loading of PDFs/images and optional ground-truth transcriptions; split management and writing intermediate artifacts.
- `pdf-to-images`: Deterministic conversion of PDF pages into images suitable for downstream processing.
- `image-preprocessing`: Configurable preprocessing (e.g., grayscale, denoise, binarize, deskew, resize) to improve downstream detection/OCR.
- `text-region-detection`: Detect and crop primary printed text regions (and optionally lines) from page images.
- `ocr-recognition`: Baseline OCR inference using a transformer OCR model (e.g., TrOCR) over detected regions/lines with reconstruction into page text.
- `llm-postcorrection`: Post-processing stage that takes OCR text and emits corrected text while preserving traceability (original vs corrected).
- `evaluation-metrics`: Compute CER/WER against available ground truth and generate a simple run report (per-page and aggregate).
- `pipeline-runner`: End-to-end and stage-wise execution via CLI/config, producing repeatable outputs into `results/`.

### Modified Capabilities

<!-- None (no existing specs in this repo yet). -->

## Impact

- New Python modules under `src/` for each pipeline stage plus shared configuration/utilities.
- New runtime dependencies expected: `torch`, `transformers`, `opencv-python`, `pdf2image`, and `jiwer` (plus any model-specific extras).
- New outputs under `results/` (intermediate images/crops, raw OCR text, corrected text, evaluation reports).

