# Brain MRI Inference Pipeline

This is a production-style inference pipeline:

`input NIfTI -> preprocess -> model -> output prediction/mask`

## Files

- `infer_pipeline.py`: production CLI (single + batch)
- `slurm_submit.py`: SLURM submission CLI (single/cohort/array)
- `brain`: top-level shorthand CLI (`./brain ...`)
- `requirements.txt`: Python dependencies
- `model_meta.example.json`: optional model validation template

## Top-level CLI (`./brain`)

You can use short commands instead of calling Python scripts directly.

### Local run

```bash
./brain run --input /path/to/sub-01.nii.gz --output-dir ./outputs --mock
```

### Submit single subject to SLURM

```bash
./brain submit sub-01 \
  --input /path/to/sub-01.nii.gz \
  --output-dir /path/to/outputs \
  --partition gpu \
  --gres gpu:1 \
  --dry-run
```

### Submit multiple subjects to SLURM

```bash
./brain submit cohort \
  --input-dir /path/to/nifti_dir \
  --output-dir /path/to/outputs \
  --partition gpu \
  --gres gpu:1
```

```bash
./brain submit array \
  --input-dir /path/to/nifti_dir \
  --output-dir /path/to/outputs \
  --partition gpu \
  --gres gpu:1 \
  --max-parallel 20
```

### Real BrainIAC segmentation (local)

```bash
./brain segment \
  --input-dir /path/to/nifti_dir \
  --output-dir /path/to/seg_outputs \
  --checkpoint-path ./checkpoints_real/segmentation.ckpt \
  --simclr-checkpoint-path ./checkpoints_real/BrainIAC.ckpt \
  --gpu-device cpu
```

### Real BrainIAC segmentation (SLURM)

```bash
./brain submit-seg cohort \
  --input-dir /path/to/nifti_dir \
  --output-dir /path/to/seg_outputs \
  --job-name brain_seg \
  --time 02:00:00 \
  --cpus 4 \
  --mem 24G
```

### Run all downstream tasks from one CLI command

```bash
./brain models \
  --input-dir /path/to/nifti_dir \
  --tasks all \
  --output-dir /path/to/all_task_outputs
```

Single-task shortcuts:

```bash
./brain age --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain idh --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain mci --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain stroke --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain sequence --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain survival --input /path/to/sub-01_T1w.nii.gz --output-dir /path/to/out
./brain seg --input /path/to/sub-01_FLAIR.nii.gz --output-dir /path/to/out
```

Combined tasks in one command:

```bash
./brain run-tasks segmentation brainage idh \
  --input-dir /path/to/nifti_dir \
  --output-dir /path/to/out
```

Run a subset:

```bash
./brain models \
  --input /path/to/sub-01_FLAIR.nii.gz \
  --tasks segmentation,sequence,brainage \
  --output-dir /path/to/sub-01_outputs
```

Submit selected/all tasks to SLURM:

```bash
./brain submit-models \
  --input-dir /path/to/nifti_dir \
  --tasks all \
  --output-dir /path/to/all_task_outputs \
  --job-prefix brain_task
```

Notes:
- `segmentation`, `brainage`, `idh`, `mci`, and `stroke` were validated end-to-end in this environment.
- `sequence` checkpoint in the current bundle appears corrupted for `torch.load` and is auto-skipped with a clear message.
- `survival` now expects a ViT-compatible checkpoint (`os.ckpt`) and will skip incompatible ResNet-style files; provide it via `--checkpoint-map survival=/path/os.ckpt`.

### Check SLURM status and logs from CLI

```bash
./brain status
./brain status --job-ids 39568,39569
./brain logs --path ./outputs_Data_Brain/pipeline.log --lines 120
./brain logs --path ./slurm_logs --lines 80
```

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run with a real model

Use a TorchScript model (`.pt`/`.pth`) that accepts `[B, C, D, H, W]` and outputs either:
- segmentation logits: `[B, C, D, H, W]`, or
- prediction logits: `[B, C]`.

```bash
python infer_pipeline.py \
  --input /path/to/subject.nii.gz \
  --model-path /path/to/model.pt \
  --output-dir ./outputs \
  --save-preprocessed
```

## 3) Batch run (cohort)

```bash
python infer_pipeline.py \
  --input-dir /path/to/nifti_dir \
  --model-path /path/to/model.pt \
  --output-dir ./outputs \
  --save-preprocessed
```

## 4) Run in mock mode (no weights needed)

This validates end-to-end IO and transform flow.

```bash
python infer_pipeline.py \
  --input /path/to/subject.nii.gz \
  --output-dir ./outputs \
  --mock \
  --save-preprocessed
```

## 5) Submit on SLURM

### Single subject job

```bash
python slurm_submit.py \
  --mode single \
  --input /path/to/subj001.nii.gz \
  --model-path /path/to/model.pt \
  --output-dir /path/to/outputs \
  --job-name brain_single \
  --partition gpu \
  --gres gpu:1 \
  --time 02:00:00 \
  --mem 16G \
  --cpus 4
```

### Multiple subjects as one cohort job

```bash
python slurm_submit.py \
  --mode cohort \
  --input-dir /path/to/nifti_dir \
  --model-path /path/to/model.pt \
  --output-dir /path/to/outputs \
  --job-name brain_cohort \
  --partition gpu \
  --gres gpu:1 \
  --time 08:00:00 \
  --mem 32G \
  --cpus 8
```

### Multiple subjects as SLURM array (recommended for scale)

```bash
python slurm_submit.py \
  --mode array \
  --input-dir /path/to/nifti_dir \
  --model-path /path/to/model.pt \
  --output-dir /path/to/outputs \
  --job-name brain_array \
  --partition gpu \
  --gres gpu:1 \
  --time 02:00:00 \
  --mem 16G \
  --cpus 4 \
  --max-parallel 20
```

### Dry-run (print sbatch command only)

```bash
python slurm_submit.py ... --dry-run
```

## Optional model metadata validation

You can pass a metadata JSON file to validate run context:

```json
{
  "expected_task": "segmentation",
  "expected_sequence": "FLAIR"
}
```

Then run:

```bash
python infer_pipeline.py \
  --input-dir /path/to/nifti_dir \
  --model-path /path/to/model.pt \
  --model-meta /path/to/model_meta.json \
  --task segmentation \
  --sequence FLAIR \
  --output-dir ./outputs
```

## Output files

Per subject, depending on model output type:
- `<subject>_mask.nii.gz` (for segmentation output)
- `<subject>_prediction.json` (for scalar/class output)
- `<subject>_preprocessed.nii.gz` (if `--save-preprocessed`)
- `<subject>_manifest.json` (subject metadata)

Per run:
- `pipeline.log`
- `run_summary_<timestamp>.json`
- `run_summary_<timestamp>.csv`

## Notes

- Default preprocessing:
  - isotropic resampling to `1.0 mm`
  - robust clipping (0.5-99.5 percentile)
  - z-score normalization on nonzero voxels
  - center pad/crop to `96x96x96`
- Use `--target-spacing` and `--target-shape` to match your trained model.
- By default batch mode continues on individual subject errors; use `--fail-fast` to stop immediately.
- This pipeline is intended for research/engineering workflows and should be validated on local data before clinical conclusions.
