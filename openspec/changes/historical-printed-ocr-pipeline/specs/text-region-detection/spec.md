## ADDED Requirements

### Requirement: Produce detected text regions
The system SHALL detect primary printed text regions on each preprocessed page image and MUST output a structured list of regions with bounding boxes in pixel coordinates and a `page_id` association.

#### Scenario: Detect regions on a page
- **WHEN** the `text-region-detection` stage runs on a preprocessed page image
- **THEN** it outputs one or more regions with bounding boxes tied to the page

### Requirement: Reading order
The system MUST order detected regions in a deterministic reading order suitable for reconstruction of page text (top-to-bottom, left-to-right with multi-column handling when configured).

#### Scenario: Two-column page is ordered
- **WHEN** a page contains two printed columns and column mode is enabled
- **THEN** regions are ordered column-by-column in reading order

### Requirement: Optional line segmentation
The system SHALL support optional line segmentation within detected regions and MUST output line-level crops and coordinates when enabled.

#### Scenario: Line segmentation enabled
- **WHEN** the configuration enables line segmentation
- **THEN** the output includes line crops and their coordinates for OCR consumption

### Requirement: Visual debug artifacts
The system SHALL generate a visualization overlay (or equivalent) showing detected regions/lines on the page to support quick quality assessment.

#### Scenario: User inspects detection quality
- **WHEN** detection completes for a page
- **THEN** a visualization artifact is available in the run directory for that page

