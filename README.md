# Brain MRI Inference Pipeline

CLI pipeline for Brain MRI inference and SLURM submission.

Flow: `input NIfTI -> preprocess -> model -> outputs`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./brain run-tasks segmentation brainage idh mci stroke sequence survival \
  --input-dir /path/to/sub-01/anat \
  --output-dir /path/to/outputs
```

Use only validated tasks:

```bash
./brain run-tasks segmentation brainage idh mci stroke \
  --input-dir /path/to/sub-01/anat \
  --output-dir /path/to/outputs
```

Submit to SLURM:

```bash
./brain submit-models \
  --input-dir /path/to/cohort \
  --tasks segmentation,brainage,idh,mci,stroke \
  --output-dir /path/to/outputs \
  --job-prefix brain_task
```

Check jobs/logs:

```bash
./brain status
./brain status --job-ids 12345,12346
./brain logs --path ./slurm_logs --lines 80
```

## Outputs

Task folders are created under `--output-dir` (for example: `segmentation`, `brainage`, `idh`, `mci`, `stroke`).
Quantitative summaries are created automatically:

- `quantitative_summary.csv`
- `quantitative_summary.json`

Included fields:

- `brainage`: `predicted_age`
- `idh`, `mci`, `stroke`: `pred_prob`, `pred_label`, `pred_logit`
- `segmentation`: `mask_nonzero_voxels`, `mask_nonzero_fraction`, `mask_shape`

## Known Limits

- Validated tasks in this environment: `segmentation`, `brainage`, `idh`, `mci`, `stroke`
- `sequence` is skipped when the checkpoint is unreadable
- `survival` requires a ViT-compatible `os.ckpt`; incompatible files are skipped

## Optional Generic Path

If you need TorchScript inference:

```bash
python infer_pipeline.py \
  --input /path/to/subject.nii.gz \
  --model-path /path/to/model.pt \
  --output-dir ./outputs \
  --save-preprocessed
```
