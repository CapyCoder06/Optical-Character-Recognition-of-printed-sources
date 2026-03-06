## ADDED Requirements

### Requirement: Deterministic PDF to image conversion
The system SHALL convert PDF pages to raster images deterministically given a configured DPI and output format, and MUST write images with stable filenames that encode document and page number.

#### Scenario: Convert a PDF at 300 DPI
- **WHEN** the user runs the `pdf-to-images` stage with `dpi=300`
- **THEN** the system writes one image per selected page with stable, predictable filenames

### Requirement: Page selection
The system MUST support converting a subset of pages (by explicit page list and/or inclusive page ranges) without converting the entire PDF.

#### Scenario: Convert only pages 1-3
- **WHEN** the configuration selects pages 1 through 3
- **THEN** the system converts only those pages and records the selection in the run metadata

### Requirement: Conversion manifest
The system SHALL produce a conversion manifest that maps each output image to its source PDF path, PDF page number, and `page_id`.

#### Scenario: Inspect image provenance
- **WHEN** an image is produced from a PDF page
- **THEN** the manifest contains the source PDF and page number for that image

### Requirement: Actionable dependency errors
If PDF conversion dependencies are missing (e.g., Poppler required by the chosen backend), the system MUST fail fast with an actionable error message describing the missing dependency and how to install/configure it.

#### Scenario: Poppler is not available
- **WHEN** the user runs PDF conversion without Poppler installed and available
- **THEN** the pipeline stops before downstream stages and prints an error describing how to fix the environment

