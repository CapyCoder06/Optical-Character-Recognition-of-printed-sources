## Context

This project targets early modern printed Spanish documents, where OCR is degraded by scan noise, warped pages, irregular layouts (marginalia, headers, multi-column), and historical spelling/typography. The repository preference is a modular Python implementation with a clear pipeline boundary between:

- PDF → images
- image preprocessing
- text region / line detection
- OCR inference
- LLM correction (post-processing only)
- evaluation (CER/WER)

Constraints and environment notes:

- Windows support is required; PDF conversion via `pdf2image` typically requires Poppler installed and on PATH.
- Ground truth may be partial or missing for many pages; evaluation must gracefully skip where unavailable.
- The LLM stage must not “modernize” historical Spanish; it should correct OCR errors while preserving original orthography as much as possible.

## Goals / Non-Goals

**Goals:**

- Provide a reproducible end-to-end pipeline runnable from a single CLI entrypoint, with stage-wise execution.
- Keep each pipeline stage isolated in its own module and make data passed between stages explicit and serializable.
- Produce inspectable artifacts (images, crops, JSON metadata, raw/corrected text, metrics) under `results/` so runs can be compared.
- Establish a baseline OCR+LLM pipeline that can be improved later (better detection, fine-tuned OCR, improved prompts) without rewriting orchestration.

**Non-Goals:**

- Training a new OCR model end-to-end in this change (fine-tuning hooks may exist, but training is not required for the first pipeline).
- Perfect layout understanding (tables, complex marginalia) in the initial baseline; detection will focus on primary printed text regions/lines.
- Any LLM usage that changes document meaning, translates text, or standardizes spelling to modern Spanish.

## Decisions

### Pipeline orchestration and artifacts

- **Decision**: Implement the pipeline as pure functions per stage plus a small runner that loads a config, executes stages, and writes artifacts to a run directory.
- **Rationale**: Keeps modules testable and allows resuming/re-running individual stages without manual notebook steps.
- **Alternatives considered**:
  - Notebook-first workflow: faster to prototype but violates reproducibility and modularity goals.
  - Workflow engines (Airflow/Prefect): powerful but excessive for a research pipeline at this stage.

### OCR baseline model choice

- **Decision**: Use a HuggingFace transformer OCR model as the baseline (e.g., TrOCR “printed” checkpoint) behind a small adapter interface.
- **Rationale**: Provides a strong baseline and a clear path for future fine-tuning; swapping models should not affect other stages.
- **Alternatives considered**:
  - Tesseract: easy to run but typically weaker on historical typography and harder to integrate with modern ML evaluation.
  - Custom CRNN+CTC: flexible but increases implementation/training scope substantially.

### Text region / line detection approach

- **Decision**: Start with deterministic OpenCV-based heuristics for baseline region detection (binarization + morphology + contour filtering), with an extension point for future ML-based layout detection.
- **Rationale**: Keeps dependencies and setup lightweight while providing reasonably strong results on many printed sources; ML detectors can be added later if needed.
- **Alternatives considered**:
  - LayoutParser/Detectron2-based models: potentially better but heavy, GPU-dependent, and domain mismatch risk for early modern prints.
  - Kraken-style segmentation: strong for historical material but introduces a separate ecosystem and training complexity.

### LLM correction constraints

- **Decision**: The LLM stage is a strict post-processing step that takes OCR text and returns corrected text with traceability; when the LLM is not available, it MUST degrade gracefully to pass-through.
- **Rationale**: Ensures the pipeline is runnable without network/API keys and preserves reproducibility; traceability reduces the risk of silent hallucinations.
- **Alternatives considered**:
  - Using LLM during recognition (image-to-text VLM): out of scope and harder to evaluate consistently.

### Configuration and interfaces

- **Decision**: Use a single config file (YAML/JSON) describing inputs, stage parameters, and output directories; expose a CLI that can override common options (input path, run name, stage selection).
- **Rationale**: Enables reproducible runs and consistent parameter tracking.
- **Alternatives considered**:
  - Hardcoded paths: explicitly disallowed.
  - Many ad-hoc CLI flags: becomes unmaintainable as the pipeline grows.

## Risks / Trade-offs

- **[PDF conversion setup friction on Windows]** → Provide clear dependency checks and a helpful error message when Poppler is missing.
- **[OpenCV heuristics fail on complex layouts]** → Keep outputs inspectable and define an interface so a future ML detector can be plugged in.
- **[LLM hallucination or unwanted modernization]** → Constrain prompts to “OCR error correction only,” require preservation of line breaks where possible, and store both raw/corrected outputs for auditing.
- **[Evaluation misleading due to partial ground truth]** → Compute metrics only where ground truth exists; report coverage (pages evaluated / total).

## Migration Plan

- Introduce new `src/` modules and a CLI runner without breaking any existing interfaces (there are no existing specs yet).
- Default behavior: running the pipeline with only PDFs/images and no labels should still produce OCR and corrected text, plus a report that notes evaluation is skipped.
- Rollback: remove the new runner/modules and keep data artifacts untouched (stored under `results/` per run).

## Open Questions

- What ground-truth format is available (per-page text files, CSV, JSONL)? The dataset I/O layer will start with a simple, explicit format and can add adapters.
- Should line segmentation be a first-class stage (for OCR accuracy) or optional within text detection? Baseline will support region detection plus optional line slicing.
- What evaluation granularity is most useful (page-level, line-level)? Baseline will include page-level with optional per-region breakdown.

