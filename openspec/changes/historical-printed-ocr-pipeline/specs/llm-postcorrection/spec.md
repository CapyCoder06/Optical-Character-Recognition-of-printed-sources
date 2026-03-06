## ADDED Requirements

### Requirement: Correction-only post-processing
The system SHALL apply an LLM as a post-processing step to correct OCR errors and MUST constrain the task to OCR correction only (no translation, summarization, or modernization of spelling beyond correcting recognition mistakes).

#### Scenario: LLM is instructed to correct OCR only
- **WHEN** the LLM correction stage runs on OCR text
- **THEN** the prompt and configuration indicate that only OCR errors are to be corrected

### Requirement: Preserve layout signals
The system MUST preserve basic layout signals from OCR output (at minimum: paragraph and line breaks when present) unless explicitly configured otherwise.

#### Scenario: Line breaks are preserved
- **WHEN** OCR output contains line breaks from line segmentation
- **THEN** the corrected text output retains line breaks in corresponding positions

### Requirement: Traceability and auditability
The system SHALL store both raw OCR text and corrected text, and MUST write metadata sufficient to audit the correction (model identifier, prompt template version, and timing), along with a simple diff or alignment artifact.

#### Scenario: User audits a correction
- **WHEN** the user inspects a corrected page
- **THEN** the run artifacts include raw text, corrected text, and metadata describing how correction was produced

### Requirement: Graceful degradation without LLM access
If the configured LLM is unavailable (missing credentials, network failure, or disabled), the system MUST degrade gracefully by producing a pass-through corrected output identical to raw OCR text and recording that correction was skipped.

#### Scenario: No LLM credentials provided
- **WHEN** the correction stage is executed without credentials
- **THEN** the system emits corrected output equal to the OCR output and marks the stage as skipped in metadata

