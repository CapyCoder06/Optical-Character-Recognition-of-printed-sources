## ADDED Requirements

### Requirement: Configurable dataset inputs
The system SHALL load all dataset inputs (PDFs, images, and optional ground truth) from an explicit configuration file and MUST validate that configured paths exist before processing begins.

#### Scenario: Run starts with valid configuration
- **WHEN** the user runs the pipeline with a configuration referencing an existing `data/` directory
- **THEN** the pipeline loads inputs without hardcoded paths and begins processing

### Requirement: Stable page identifiers
The system SHALL assign each processed page a stable `page_id` derived deterministically from the source document name and page number, and MUST persist a manifest mapping `page_id` to original source.

#### Scenario: Re-running produces identical page identifiers
- **WHEN** the same PDF is processed twice with the same page selection
- **THEN** the resulting `page_id` values match across runs

### Requirement: Run-scoped artifact outputs
The system SHALL write outputs under a run directory (e.g., `results/<run_id>/`) and MUST store a machine-readable manifest of produced artifacts (images, crops, text, metrics) for traceability.

#### Scenario: Artifacts are discoverable for a run
- **WHEN** a run completes
- **THEN** the run directory contains a manifest that lists all produced files and their associated `page_id`

### Requirement: Optional ground truth support
The system MUST allow runs without ground-truth transcriptions and SHALL report evaluation coverage (how many pages have ground truth) when metrics are computed.

#### Scenario: Dataset lacks labels
- **WHEN** the dataset configuration omits ground-truth paths
- **THEN** the pipeline still produces OCR and corrected text and reports that evaluation was skipped due to missing ground truth

