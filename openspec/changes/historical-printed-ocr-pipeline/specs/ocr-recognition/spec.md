## ADDED Requirements

### Requirement: Model-backed OCR inference
The system SHALL perform OCR inference over detected regions or lines using a configured transformer OCR model and MUST allow selecting the model checkpoint via configuration.

#### Scenario: Run OCR with a configured model
- **WHEN** the configuration specifies an OCR model checkpoint
- **THEN** the OCR stage loads that checkpoint and produces text for each region/line

### Requirement: Per-item OCR outputs with confidence
The system MUST produce per-region/per-line OCR outputs that include recognized text and a confidence signal (or a documented proxy) suitable for downstream correction and debugging.

#### Scenario: Inspect uncertain lines
- **WHEN** OCR completes for a page
- **THEN** the output includes confidences that allow identifying low-confidence regions/lines

### Requirement: Page text reconstruction
The system SHALL reconstruct page-level text by joining region/line outputs according to the reading order emitted by text detection, preserving line breaks when line segmentation is enabled.

#### Scenario: Page text is reconstructed
- **WHEN** OCR is run with line segmentation enabled
- **THEN** the page-level text output contains line breaks consistent with detected line ordering

### Requirement: Device selection and repeatability
The system MUST allow selecting the inference device (CPU/GPU when available) and SHALL record model and device details in run metadata for repeatability.

#### Scenario: Compare two runs
- **WHEN** the same page is processed on CPU vs GPU
- **THEN** the run metadata records device choice so results can be compared and interpreted

