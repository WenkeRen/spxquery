# SED Integration Recommendations

The new `sed/` module is feature complete on its own, but much of its logic overlaps with the existing SPXQuery pipeline that lives under `core/`, `processing/`, and `visualization/`. The notes below summarize the current structure and list concrete adjustments that will make the SWT-based reconstruction behave like a first-class pipeline stage instead of a parallel prototype.

## Current Architecture Snapshot

- **Pipeline orchestration**: `core/pipeline.py` already defines resumable stages (`query → download → processing → visualization`) backed by `AdvancedConfig` in `core/config.py`.
- **Processing outputs**: `run_processing()` writes `results/lightcurve.csv` via `processing.lightcurve.generate_lightcurve_dataframe/save_lightcurve_csv`, and records state/paths on `PipelineState`.
- **SED module**: `sed/reconstruction.py`, `sed/data_loader.py`, `sed/matrices.py`, etc. expect to read a light-curve CSV directly, apply their own QC filters, and emit spectra/diagnostics in an arbitrary output directory.
- **Shared helpers**: `utils/helpers.py` already contains flag/SNR filters (`apply_quality_filters`, `classify_photometry_by_quality`) plus serialization utilities that the SED code re-implements.

## Suggested Adjustments

### 1. Add an explicit "sed" pipeline stage

- Extend `SPXQueryPipeline.STAGE_DEPENDENCIES` and `pipeline_stages` defaults to include a final `"sed"` stage that depends on `processing`.
- Implement `run_sed()` alongside `run_processing()` so it can pull `self.state.csv_path` and call `SEDReconstructor` once photometry finishes.
- Persist SED artifacts (CSV, YAML, plots) under `results/sed/` and append summary info to the saved pipeline state so downstream notebooks know whether reconstruction already ran.

### 2. Route SED parameters through `AdvancedConfig`

- Introduce an optional `sed: SEDConfig` field on `AdvancedConfig` (`core/config.py`) and teach `AdvancedConfig.update()` how to fan out the SWT-specific kwargs.
- Accept a `sed_config_path` in the CLI / YAML so users can tweak one config tree instead of juggling `sed_config.yaml` separately.
- When `run_sed()` executes, pull `config.sed` (or build a default) so all hyperparameters live beside query/photometry/download values in the persisted `{source}.yaml` state file.

### 3. Reuse the canonical light-curve loaders

- Replace `sed.data_loader.load_lightcurve_csv` with a thin wrapper around `processing.lightcurve.load_lightcurve_from_csv` so there is only one CSV schema definition.
- Rather than rebuilding pandas frames from scratch, consider feeding `List[PhotometryResult]` into the reconstruction layer and using lightweight translators (e.g., a helper that converts to the `BandData` arrays currently required by `build_all_matrices`).
- This avoids divergent column requirements and lets future pipeline changes (new metadata columns, unit conversions) flow automatically into SED.

### 4. Consolidate quality-control logic

- `sed.data_loader.apply_quality_filters` duplicates and extends `utils.helpers.apply_quality_filters` (bad-flag masking, SNR cuts, rolling MAD clipping).
- Move the sigma-clipping and rejection bookkeeping into `utils.helpers` so both the photometry visualizer and the SED module consume the same implementation.
- Expose the richer rejection stats (per reason counts) on the helper so `processing` logs gain this detail "for free" while SED loses bespoke filtering code.

### 5. Align outputs and metadata handling

- `SEDReconstructionResult.save_all()` currently writes wherever the caller points it; standardize on `results/sed/` relative to the pipeline output dir and register those paths on `PipelineState` (e.g., `state.sed_csv_path`, `state.sed_yaml_path`).
- Save summary diagnostics (per-band reduced χ², tuning choices) into `results/sed/summary.yaml` and optionally append a small entry into `results/query_summary.yaml` so users can see at a glance whether SED succeeded.
- Reuse `utils.helpers.save_yaml/load_yaml` for these artifacts to match the rest of the codebase and reduce bespoke YAML handling inside `sed/reconstruction.py`.

### 6. Hook SED plots into the visualization package

- `sed/plots.py` builds its own Matplotlib figures even though `visualization/plots.py` already manages style defaults, sigma clipping toggles, etc.
- Export the band comparison and diagnostic plots via the existing visualization package (e.g., add a `sed` submodule under `visualization/` or register plotting functions that `run_visualization()` can call when SED results exist).
- Share color maps and helper utilities (smart y-limits, upper-limit glyphs) to keep a consistent look between light-curve plots and SED panels.

### 7. Testing & documentation follow-up

- Add integration tests under `tests/` that execute `run_sed()` against the existing `test_lightcurve.csv`, asserting that spectra/metadata files are produced and contain expected keys.
- Expand `docs/user_guide/` with a "SED stage" page that explains required config knobs, how to enable/disable the stage, and where outputs land.
- Update `README.md`/`INSTALL.md` dependency sections so CVXPY/pywt requirements are called out for users running the full pipeline.

Implementing the items above keeps the SED reconstruction powerful while eliminating duplicate loaders/configuration paths, letting users trigger everything through the same SPXQuery pipeline experience.
