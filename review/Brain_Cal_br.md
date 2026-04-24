# Builder Review — BrainIAC paper + Brain_Cal implementation

**Primary reference:** BrainIAC paper (`brain_model_paper.pdf`).

## Context

This review assesses the paper from a builder perspective using the implemented `Brain_Cal` pipeline in this repo.  
Focus is on deployment realism: modality handling, checkpoint compatibility, execution reliability, and output interpretability.

Implemented surface:

- Local execution: `./brain models`, `./brain run-tasks`
- SLURM execution: `./brain submit-models`
- Structured outputs: per-task JSON artifacts + `quantitative_summary.csv/json`

## Usability and reproducibility

### What is strong

- Single CLI unifies multiple downstream tasks in one place.
- Runtime checks are explicit: missing/corrupted/incompatible checkpoints are skipped with clear messages.
- Quantitative summaries are auto-generated, reducing manual aggregation.
- Brain-age export now includes both `predicted_age_months` and `predicted_age_years`, which improves interpretability.

### What was fixed in this build

- **Modality gating for brain-age:** run only on T1-like files (for example `*_T1w.nii.gz`), preventing accidental FLAIR-only runs.
- **Dual-input handling for IDH:** requires paired `*_t2f` and `*_t1ce` files for local task execution.
- **Safer task orchestration:** task-level skip behavior avoids full-pipeline failure when one model is unavailable.

Builder view: these changes materially improve reproducibility of *what is actually being run* versus what the paper task definitions imply.

## Observed performance in the real run

Source: `outputs_sub01_real_updated/quantitative_summary.csv`

- `brainage` on T1w: `238.69` months (`19.89` years)
- `mci`: low probabilities, label `0` on both available images
- `stroke`: very high probabilities, label `1` on both available images
- `segmentation`: valid mask metrics (nonzero voxel counts/fractions) for T1w and FLAIR

Interpretation from a builder lens:

- The pipeline runs end-to-end for available validated tasks.
- Numeric outputs are produced in a consistent format suitable for downstream review/reporting.
- Classification probabilities remain model scores; calibration and external validation are still needed before stronger claims.

## Generalization and deployment risk

- Site/scanner/protocol shifts can affect output behavior; paper performance should be treated as a prior, not a guarantee.
- Modality mismatch is a common failure mode in real deployments; recent gating reduces this risk.
- Checkpoint architecture mismatch (for example ResNet-style weights with ViT loaders) is now detected earlier and handled safely.
- Overconfident probabilities (for example values near `1.0`) are possible and should trigger calibration/QC review, not automatic acceptance.

## Integration readiness

Current status:

- Practical for research deployment on local machines and SLURM.
- Good artifact structure for cohort-scale bookkeeping and auditability.

Remaining engineering gaps:

- `sequence` task blocked by corrupted checkpoint file.
- `survival` task blocked by missing ViT-compatible `os.ckpt`.
- `submit-models` currently skips IDH until paired-input SLURM wrapping is implemented.

## Builder conclusion

The paper is implementable in practice, and this repository now has core production controls that matter in real operations: modality-aware routing, checkpoint validation, and structured quantitative outputs.  
The main blockers are now external assets and last-mile orchestration (missing/incompatible checkpoints and paired-IDH SLURM path), not the overall pipeline design.

*Last updated: 2026-03-05.*
