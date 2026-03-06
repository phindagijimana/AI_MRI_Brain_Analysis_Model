#!/usr/bin/env python3
"""
Production-ready brain MRI inference pipeline.

Pipeline:
    input NIfTI -> preprocess -> model -> output prediction/mask

Features:
  - Single-subject and batch mode.
  - Structured logging and run summaries.
  - Optional model metadata validation (expected task/sequence).
  - Safe per-subject failure handling for cohort runs.
  - Mock inference mode for end-to-end smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch


LOGGER = logging.getLogger("brain_pipeline")
VALID_TASKS = {"auto", "segmentation", "classification", "regression"}
VALID_SEQUENCES = {"unknown", "T1", "T2", "FLAIR", "T1CE"}


@dataclass
class PreprocessMeta:
    original_shape: Tuple[int, int, int]
    resampled_shape: Tuple[int, int, int]
    original_zooms: Tuple[float, float, float]
    target_spacing: float
    src_slices: Tuple[slice, slice, slice]
    dst_slices: Tuple[slice, slice, slice]


@dataclass
class RunConfig:
    output_dir: Path
    model_path: Optional[Path]
    model_meta_path: Optional[Path]
    task: str
    sequence: str
    target_spacing: float
    target_shape: Tuple[int, int, int]
    device: str
    save_preprocessed: bool
    mock: bool
    batch_mode: bool
    fail_fast: bool
    seed: int


def configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    LOGGER.setLevel(level)
    LOGGER.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    LOGGER.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_subject_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
    elif name.endswith(".nii"):
        base = name[:-4]
    else:
        base = path.stem
    # Keep ids file-safe and deterministic.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _ensure_3d(array: np.ndarray) -> np.ndarray:
    if array.ndim > 3:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape={array.shape}")
    return array


def _center_pad_or_crop(
    image: np.ndarray, target_shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
    out = np.zeros(target_shape, dtype=image.dtype)
    src_slices: List[slice] = []
    dst_slices: List[slice] = []

    for dim in range(3):
        in_len = image.shape[dim]
        out_len = target_shape[dim]
        if in_len >= out_len:
            start = (in_len - out_len) // 2
            src = slice(start, start + out_len)
            dst = slice(0, out_len)
        else:
            pad_start = (out_len - in_len) // 2
            src = slice(0, in_len)
            dst = slice(pad_start, pad_start + in_len)
        src_slices.append(src)
        dst_slices.append(dst)

    src_t = tuple(src_slices)
    dst_t = tuple(dst_slices)
    out[dst_t] = image[src_t]
    return out, src_t, dst_t


def preprocess_nifti(
    nifti_path: Path,
    target_spacing: float = 1.0,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
) -> Tuple[np.ndarray, nib.Nifti1Image, PreprocessMeta]:
    nii = nib.load(str(nifti_path))
    data = nii.get_fdata(dtype=np.float32)
    data = _ensure_3d(data)
    data = np.nan_to_num(data, copy=False)

    original_shape = tuple(int(x) for x in data.shape)
    original_zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
    if any(z <= 0 for z in original_zooms):
        raise ValueError(f"Invalid voxel spacing in header: {original_zooms}")

    zoom_factors = tuple(z / target_spacing for z in original_zooms)
    resampled = ndi.zoom(data, zoom=zoom_factors, order=1)
    resampled_shape = tuple(int(x) for x in resampled.shape)

    p_low, p_high = np.percentile(resampled, [0.5, 99.5])
    resampled = np.clip(resampled, p_low, p_high)
    nonzero = resampled != 0
    if np.any(nonzero):
        mean = float(resampled[nonzero].mean())
        std = float(resampled[nonzero].std())
        resampled = (resampled - mean) / (std if std > 1e-6 else 1.0)
    else:
        resampled = np.zeros_like(resampled, dtype=np.float32)

    proc, src_slices, dst_slices = _center_pad_or_crop(resampled, target_shape)
    proc = proc.astype(np.float32)

    meta = PreprocessMeta(
        original_shape=original_shape,
        resampled_shape=resampled_shape,
        original_zooms=original_zooms,
        target_spacing=float(target_spacing),
        src_slices=src_slices,
        dst_slices=dst_slices,
    )
    return proc, nii, meta


def invert_mask_to_original_space(mask_proc: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    resampled_mask = np.zeros(meta.resampled_shape, dtype=np.uint8)
    resampled_mask[meta.src_slices] = mask_proc[meta.dst_slices].astype(np.uint8)

    zoom_back = tuple(
        float(meta.original_shape[i]) / float(meta.resampled_shape[i]) for i in range(3)
    )
    restored = ndi.zoom(resampled_mask.astype(np.float32), zoom=zoom_back, order=0).astype(np.uint8)

    if restored.shape != meta.original_shape:
        fixed = np.zeros(meta.original_shape, dtype=np.uint8)
        min_shape = tuple(min(restored.shape[d], meta.original_shape[d]) for d in range(3))
        fixed[0:min_shape[0], 0:min_shape[1], 0:min_shape[2]] = restored[
            0:min_shape[0], 0:min_shape[1], 0:min_shape[2]
        ]
        restored = fixed
    return restored


def _extract_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("Model output has no tensor payload.")


def load_model_metadata(meta_path: Optional[Path]) -> Dict[str, Any]:
    if meta_path is None:
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Model metadata JSON must be an object.")
    return data


def run_model(
    image_proc: np.ndarray,
    model_path: Optional[Path],
    device: str,
    mock: bool,
) -> Dict[str, Any]:
    x = torch.from_numpy(image_proc).unsqueeze(0).unsqueeze(0).to(device)  # [B,C,D,H,W]

    if mock:
        z = image_proc
        pseudo_mask = (z > 0.75).astype(np.uint8)
        score = float(np.clip((z.mean() + 2.0) / 4.0, 0.0, 1.0))
        return {"mode": "mock", "mask": pseudo_mask, "prediction": {"score": score}}

    if model_path is None:
        raise ValueError("A model file is required unless --mock is set.")

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    with torch.no_grad():
        out_raw = model(x)
    out = _extract_tensor_output(out_raw)

    if out.ndim == 5:
        logits = out[0]
        if logits.shape[0] > 1:
            mask = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
        else:
            prob = torch.sigmoid(logits[0])
            mask = (prob > 0.5).cpu().numpy().astype(np.uint8)
        return {"mode": "segmentation", "mask": mask}

    if out.ndim == 2:
        vec = out[0].float()
        if vec.numel() > 1:
            probs = torch.softmax(vec, dim=0).cpu().numpy().tolist()
            pred_idx = int(np.argmax(probs))
            return {"mode": "prediction", "prediction": {"class_index": pred_idx, "probabilities": probs}}
        score = float(torch.sigmoid(vec[0]).cpu().item())
        return {"mode": "prediction", "prediction": {"score": score}}

    raise ValueError(
        f"Unsupported model output shape {tuple(out.shape)}. Expected [B,C,D,H,W] or [B,C]."
    )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def save_outputs(
    output_dir: Path,
    subject_id: str,
    input_path: Path,
    original_nii: nib.Nifti1Image,
    meta: PreprocessMeta,
    proc_image: np.ndarray,
    results: Dict[str, Any],
    save_preprocessed: bool,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: List[str] = []

    if save_preprocessed:
        proc_name = f"{subject_id}_preprocessed.nii.gz"
        proc_path = output_dir / proc_name
        nib.save(nib.Nifti1Image(proc_image.astype(np.float32), np.eye(4)), str(proc_path))
        produced.append(proc_name)

    if "mask" in results:
        mask_proc = results["mask"].astype(np.uint8)
        mask_orig = invert_mask_to_original_space(mask_proc, meta)
        mask_name = f"{subject_id}_mask.nii.gz"
        mask_path = output_dir / mask_name
        nib.save(nib.Nifti1Image(mask_orig, affine=original_nii.affine, header=original_nii.header), str(mask_path))
        produced.append(mask_name)

    if "prediction" in results:
        pred_name = f"{subject_id}_prediction.json"
        pred_path = output_dir / pred_name
        atomic_write_json(pred_path, results["prediction"])
        produced.append(pred_name)

    manifest = {
        "subject_id": subject_id,
        "input_path": str(input_path),
        "original_shape": meta.original_shape,
        "resampled_shape": meta.resampled_shape,
        "target_spacing": meta.target_spacing,
        "mode": results.get("mode", "unknown"),
        "outputs": produced,
    }
    manifest_name = f"{subject_id}_manifest.json"
    atomic_write_json(output_dir / manifest_name, manifest)
    produced.append(manifest_name)
    return produced


def validate_config(cfg: RunConfig, model_meta: Dict[str, Any]) -> None:
    if cfg.task not in VALID_TASKS:
        raise ValueError(f"Invalid --task '{cfg.task}'. Allowed: {sorted(VALID_TASKS)}")
    if cfg.sequence not in VALID_SEQUENCES:
        raise ValueError(f"Invalid --sequence '{cfg.sequence}'. Allowed: {sorted(VALID_SEQUENCES)}")
    if cfg.target_spacing <= 0:
        raise ValueError("--target-spacing must be > 0")
    if any(dim <= 0 for dim in cfg.target_shape):
        raise ValueError("--target-shape dimensions must be > 0")

    if not cfg.mock and cfg.model_path is None:
        raise ValueError("Provide --model-path unless --mock is set.")
    if cfg.model_path is not None and not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    expected_task = model_meta.get("expected_task")
    expected_sequence = model_meta.get("expected_sequence")
    if expected_task and cfg.task != "auto" and cfg.task != expected_task:
        raise ValueError(
            f"Task mismatch: CLI --task={cfg.task} but model metadata expected_task={expected_task}"
        )
    if expected_sequence and cfg.sequence != "unknown" and cfg.sequence != expected_sequence:
        raise ValueError(
            "Sequence mismatch: CLI --sequence="
            f"{cfg.sequence} but model metadata expected_sequence={expected_sequence}"
        )


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def discover_inputs(single_input: Optional[Path], input_dir: Optional[Path]) -> List[Path]:
    if single_input is not None and input_dir is not None:
        raise ValueError("Use either --input or --input-dir, not both.")
    if single_input is None and input_dir is None:
        raise ValueError("Provide one of --input or --input-dir.")

    if single_input is not None:
        if not single_input.exists():
            raise FileNotFoundError(f"Input not found: {single_input}")
        return [single_input]

    assert input_dir is not None
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = sorted(list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz")))
    if not files:
        raise ValueError(f"No NIfTI files found in: {input_dir}")
    return files


def run_one_subject(input_path: Path, cfg: RunConfig) -> Dict[str, Any]:
    subject_id = safe_subject_id(input_path)
    t0 = time.time()
    proc_img, original_nii, meta = preprocess_nifti(
        nifti_path=input_path,
        target_spacing=cfg.target_spacing,
        target_shape=cfg.target_shape,
    )
    results = run_model(proc_img, cfg.model_path, device=cfg.device, mock=cfg.mock)
    outputs = save_outputs(
        output_dir=cfg.output_dir,
        subject_id=subject_id,
        input_path=input_path,
        original_nii=original_nii,
        meta=meta,
        proc_image=proc_img,
        results=results,
        save_preprocessed=cfg.save_preprocessed,
    )
    elapsed_s = round(time.time() - t0, 3)
    return {
        "subject_id": subject_id,
        "input_path": str(input_path),
        "status": "ok",
        "mode": results.get("mode", "unknown"),
        "outputs": outputs,
        "elapsed_s": elapsed_s,
    }


def write_run_summary(output_dir: Path, cfg: RunConfig, rows: Sequence[Dict[str, Any]]) -> None:
    timestamp = int(time.time())
    summary_json = output_dir / f"run_summary_{timestamp}.json"
    summary_csv = output_dir / f"run_summary_{timestamp}.csv"

    payload = {
        "config": {
            **asdict(cfg),
            "output_dir": str(cfg.output_dir),
            "model_path": str(cfg.model_path) if cfg.model_path else None,
            "model_meta_path": str(cfg.model_meta_path) if cfg.model_meta_path else None,
        },
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "failed": sum(1 for r in rows if r.get("status") == "failed"),
        "results": list(rows),
    }
    atomic_write_json(summary_json, payload)

    fields = ["subject_id", "input_path", "status", "mode", "elapsed_s", "error"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject_id": row.get("subject_id"),
                    "input_path": row.get("input_path"),
                    "status": row.get("status"),
                    "mode": row.get("mode", ""),
                    "elapsed_s": row.get("elapsed_s", ""),
                    "error": row.get("error", ""),
                }
            )

    LOGGER.info("Run summary JSON: %s", summary_json)
    LOGGER.info("Run summary CSV: %s", summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production brain MRI inference: NIfTI -> preprocess -> model -> output."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single input NIfTI (.nii/.nii.gz).")
    src.add_argument("--input-dir", type=Path, help="Directory containing NIfTI files for batch run.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save outputs.")
    parser.add_argument("--model-path", type=Path, default=None, help="TorchScript model path (.pt/.pth).")
    parser.add_argument(
        "--model-meta",
        type=Path,
        default=None,
        help="Optional model metadata JSON with keys expected_task and expected_sequence.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=sorted(VALID_TASKS),
        help="Expected task mode. Uses output-shape auto detection by default.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="unknown",
        choices=sorted(VALID_SEQUENCES),
        help="Sequence label for validation/documentation.",
    )
    parser.add_argument("--target-spacing", type=float, default=1.0, help="Isotropic target spacing in mm.")
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        metavar=("D", "H", "W"),
        help="Target shape after center pad/crop.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed volume.")
    parser.add_argument("--mock", action="store_true", help="Run mock inference without model weights.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch immediately on first failure. Default is continue-on-error.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir, verbose=bool(args.verbose))
    set_seed(int(args.seed))

    cfg = RunConfig(
        output_dir=args.output_dir,
        model_path=args.model_path,
        model_meta_path=args.model_meta,
        task=args.task,
        sequence=args.sequence,
        target_spacing=float(args.target_spacing),
        target_shape=tuple(args.target_shape),
        device=resolve_device(args.device),
        save_preprocessed=bool(args.save_preprocessed),
        mock=bool(args.mock),
        batch_mode=bool(args.input_dir),
        fail_fast=bool(args.fail_fast),
        seed=int(args.seed),
    )
    model_meta = load_model_metadata(cfg.model_meta_path)
    validate_config(cfg, model_meta)

    inputs = discover_inputs(args.input, args.input_dir)
    LOGGER.info("Starting run | inputs=%d | device=%s | mock=%s", len(inputs), cfg.device, cfg.mock)

    rows: List[Dict[str, Any]] = []
    for idx, input_path in enumerate(inputs, start=1):
        LOGGER.info("Processing %d/%d: %s", idx, len(inputs), input_path)
        try:
            row = run_one_subject(input_path, cfg)
            rows.append(row)
            LOGGER.info("Completed %s in %.3fs", row["subject_id"], row["elapsed_s"])
        except Exception as exc:
            err = {
                "subject_id": safe_subject_id(input_path),
                "input_path": str(input_path),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(err)
            LOGGER.exception("Failed on %s", input_path)
            if cfg.fail_fast:
                break

    write_run_summary(cfg.output_dir, cfg, rows)
    failed = sum(1 for r in rows if r.get("status") == "failed")
    if failed:
        raise SystemExit(2)
    LOGGER.info("Done. All subjects completed successfully.")


if __name__ == "__main__":
    main()
