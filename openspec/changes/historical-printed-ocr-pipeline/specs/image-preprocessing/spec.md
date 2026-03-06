## ADDED Requirements

### Requirement: Config-driven preprocessing pipeline
The system SHALL apply a preprocessing pipeline to each input page image based on configuration, and each preprocessing step MUST be individually enableable/disableable.

#### Scenario: Enable binarization and deskew
- **WHEN** the configuration enables binarization and deskew steps
- **THEN** the system outputs a preprocessed image reflecting those steps

### Requirement: Preserve originals and record parameters
The system MUST preserve the original (unmodified) input image and SHALL record preprocessing parameters (and step order) used to generate the preprocessed image for each run.

#### Scenario: Audit preprocessing choices
- **WHEN** the user inspects a completed run
- **THEN** the run contains both original and preprocessed images and metadata describing preprocessing parameters

### Requirement: Consistent output image format
The system SHALL output preprocessed images in a consistent, documented format (color space, bit depth) suitable for downstream text detection and OCR.

#### Scenario: Downstream stages can consume outputs
- **WHEN** text detection runs after preprocessing
- **THEN** it can load and process all preprocessed images without format-related failures

### Requirement: Deterministic processing
Given the same input image and preprocessing configuration, the system MUST produce identical preprocessed output (byte-for-byte) or document any unavoidable nondeterminism.

#### Scenario: Re-run preprocessing with the same config
- **WHEN** preprocessing is executed twice on the same input with the same configuration
- **THEN** the produced preprocessed image is identical across runs

