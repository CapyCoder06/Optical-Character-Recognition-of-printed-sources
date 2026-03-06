## ADDED Requirements

### Requirement: CLI supports end-to-end and stage-wise runs
The system SHALL provide a CLI that can run the full pipeline end-to-end and MUST support running individual stages (e.g., preprocessing only, OCR only) based on user selection.

#### Scenario: Run only OCR stage
- **WHEN** the user invokes the CLI selecting only the OCR stage
- **THEN** the system runs OCR using existing upstream artifacts (or fails with an actionable message if they are missing)

### Requirement: Configuration snapshotting
The system MUST accept a configuration file as input and SHALL write an immutable snapshot of the effective configuration into the run directory for reproducibility.

#### Scenario: Record the run configuration
- **WHEN** a run starts
- **THEN** the run directory includes the effective configuration used for that run

### Requirement: Run identification and output structure
The system MUST create a unique `run_id` for each execution (user-specified or auto-generated) and SHALL write all outputs under `results/<run_id>/` following a documented folder structure.

#### Scenario: Two runs do not collide
- **WHEN** the user executes two pipeline runs with different run IDs
- **THEN** outputs are written to separate run directories without overwriting each other

### Requirement: Resume and overwrite controls
The system SHALL support resuming from existing artifacts (skipping completed stages) and MUST provide an explicit overwrite/force option to recompute artifacts when requested.

#### Scenario: Resume an interrupted run
- **WHEN** a run is re-launched with the same `run_id` and resume enabled
- **THEN** the system skips stages with completed artifacts and continues from the next required stage

