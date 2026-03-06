## ADDED Requirements

### Requirement: Compute CER and WER
When ground truth is available, the system SHALL compute Character Error Rate (CER) and Word Error Rate (WER) for OCR output and for LLM-corrected output.

#### Scenario: Evaluate a labeled page
- **WHEN** a page has an associated ground-truth transcription
- **THEN** the evaluation stage computes CER/WER for raw OCR and corrected text

### Requirement: Coverage-aware reporting
The system MUST report evaluation coverage (number of pages evaluated vs total processed) and SHALL skip metric computation for pages without ground truth without failing the run.

#### Scenario: Partial labels in dataset
- **WHEN** only some pages have ground-truth transcriptions
- **THEN** the report includes coverage and metrics are computed only for labeled pages

### Requirement: Reproducible normalization
The system SHALL apply a documented, configuration-driven normalization policy (e.g., whitespace normalization) consistently across predictions and ground truth before metric computation.

#### Scenario: Whitespace normalization enabled
- **WHEN** normalization is enabled in configuration
- **THEN** the evaluation applies the same normalization rules to both ground truth and predictions

### Requirement: Machine-readable and human-readable reports
The system MUST output a machine-readable metrics report (JSON) and SHALL produce a human-readable summary (e.g., Markdown) including aggregate and per-page results.

#### Scenario: Inspect evaluation results
- **WHEN** evaluation completes for a run
- **THEN** the run directory contains both JSON and Markdown summaries of CER/WER

