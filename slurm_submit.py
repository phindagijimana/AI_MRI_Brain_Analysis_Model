#!/usr/bin/env python3
"""
Submit brain MRI inference jobs to SLURM.

Supports:
  - Single subject submission.
  - Multi-subject submission as one cohort job.
  - Multi-subject submission as a SLURM job array (one subject per task).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


def discover_niftis(input_dir: Path) -> List[Path]:
    files = sorted(list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz")))
    return files


def q(items: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in items)


def build_common_sbatch_args(args: argparse.Namespace, logs_dir: Path) -> List[str]:
    sbatch = [
        "sbatch",
        "--job-name",
        args.job_name,
        "--output",
        str(logs_dir / "%x_%j.out"),
        "--error",
        str(logs_dir / "%x_%j.err"),
        "--time",
        args.time,
        "--cpus-per-task",
        str(args.cpus),
        "--mem",
        args.mem,
    ]
    if args.partition:
        sbatch += ["--partition", args.partition]
    if args.account:
        sbatch += ["--account", args.account]
    if args.qos:
        sbatch += ["--qos", args.qos]
    if args.gres:
        sbatch += ["--gres", args.gres]
    if args.constraint:
        sbatch += ["--constraint", args.constraint]
    if args.chdir:
        sbatch += ["--chdir", str(args.chdir)]
    return sbatch


def build_pipeline_base_cmd(args: argparse.Namespace, pipeline_path: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(pipeline_path),
        "--output-dir",
        str(args.output_dir),
        "--task",
        args.task,
        "--sequence",
        args.sequence,
        "--target-spacing",
        str(args.target_spacing),
        "--target-shape",
        str(args.target_shape[0]),
        str(args.target_shape[1]),
        str(args.target_shape[2]),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.model_path:
        cmd += ["--model-path", str(args.model_path)]
    if args.model_meta:
        cmd += ["--model-meta", str(args.model_meta)]
    if args.save_preprocessed:
        cmd += ["--save-preprocessed"]
    if args.mock:
        cmd += ["--mock"]
    if args.fail_fast:
        cmd += ["--fail-fast"]
    if args.verbose:
        cmd += ["--verbose"]
    return cmd


def run_or_print(cmd: List[str], dry_run: bool) -> int:
    print("$ " + q(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def submit_single(args: argparse.Namespace, pipeline_path: Path, logs_dir: Path) -> int:
    if args.input is None:
        raise ValueError("--input is required for single mode.")
    sbatch = build_common_sbatch_args(args, logs_dir)
    pipeline_cmd = build_pipeline_base_cmd(args, pipeline_path) + ["--input", str(args.input)]
    sbatch += ["--wrap", q(pipeline_cmd)]
    return run_or_print(sbatch, args.dry_run)


def submit_cohort(args: argparse.Namespace, pipeline_path: Path, logs_dir: Path) -> int:
    if args.input_dir is None:
        raise ValueError("--input-dir is required for cohort mode.")
    sbatch = build_common_sbatch_args(args, logs_dir)
    pipeline_cmd = build_pipeline_base_cmd(args, pipeline_path) + ["--input-dir", str(args.input_dir)]
    sbatch += ["--wrap", q(pipeline_cmd)]
    return run_or_print(sbatch, args.dry_run)


def submit_array(args: argparse.Namespace, pipeline_path: Path, logs_dir: Path) -> int:
    if args.input_dir is None:
        raise ValueError("--input-dir is required for array mode.")
    inputs = discover_niftis(args.input_dir)
    if not inputs:
        raise ValueError(f"No NIfTI files found in {args.input_dir}")

    list_path = logs_dir / f"array_inputs_{int(time.time())}.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for p in inputs:
            f.write(str(p) + "\n")

    sbatch = build_common_sbatch_args(args, logs_dir)
    sbatch[sbatch.index("--output") + 1] = str(logs_dir / "%x_%A_%a.out")
    sbatch[sbatch.index("--error") + 1] = str(logs_dir / "%x_%A_%a.err")

    max_parallel = args.max_parallel if args.max_parallel > 0 else len(inputs)
    sbatch += ["--array", f"0-{len(inputs)-1}%{max_parallel}"]

    base = build_pipeline_base_cmd(args, pipeline_path)
    # Read one file per array index; SLURM_ARRAY_TASK_ID starts at 0 here.
    body = (
        "set -euo pipefail; "
        f'INPUT_FILE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {shlex.quote(str(list_path))})"; '
        + q(base + ["--input", "$INPUT_FILE"])
    )
    sbatch += ["--wrap", f"bash -lc {shlex.quote(body)}"]
    print(f"Discovered {len(inputs)} subjects for array mode.")
    return run_or_print(sbatch, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLURM submission CLI for infer_pipeline.py")
    parser.add_argument(
        "--mode",
        choices=["single", "cohort", "array"],
        required=True,
        help="single: one subject; cohort: one job for input-dir; array: one subject per SLURM task",
    )
    parser.add_argument("--input", type=Path, help="Input NIfTI for single mode.")
    parser.add_argument("--input-dir", type=Path, help="Input directory for cohort/array modes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline output directory.")

    # Model/pipeline options.
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-meta", type=Path, default=None)
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "segmentation", "classification", "regression"])
    parser.add_argument("--sequence", type=str, default="unknown", choices=["unknown", "T1", "T2", "FLAIR", "T1CE"])
    parser.add_argument("--target-spacing", type=float, default=1.0)
    parser.add_argument("--target-shape", type=int, nargs=3, default=(96, 96, 96), metavar=("D", "H", "W"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # SLURM options.
    parser.add_argument("--job-name", type=str, default="brain_infer")
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--time", type=str, default="02:00:00")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--gres", type=str, default=None, help='Example: "gpu:1"')
    parser.add_argument("--constraint", type=str, default=None)
    parser.add_argument("--max-parallel", type=int, default=10, help="Only used in array mode.")
    parser.add_argument("--chdir", type=Path, default=None, help="Working directory for sbatch job.")
    parser.add_argument("--logs-dir", type=Path, default=Path("./slurm_logs"), help="Directory for SLURM stdout/stderr.")

    # Runtime behavior.
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable used inside SLURM job.")
    parser.add_argument("--pipeline-path", type=Path, default=Path(__file__).parent / "infer_pipeline.py")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch command without submitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {args.pipeline_path}")
    if args.model_path and not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if args.model_meta and not args.model_meta.exists():
        raise FileNotFoundError(f"Model metadata not found: {args.model_meta}")

    if args.mode == "single":
        code = submit_single(args, args.pipeline_path, args.logs_dir)
    elif args.mode == "cohort":
        code = submit_cohort(args, args.pipeline_path, args.logs_dir)
    else:
        code = submit_array(args, args.pipeline_path, args.logs_dir)

    raise SystemExit(code)


if __name__ == "__main__":
    main()
