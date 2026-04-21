import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DETECTION_EXPERIMENT = REPO_ROOT / "work_dirs/detection/coco2017_fasterrcnn_dinov3_local_interaction"
DEFAULT_SEGMENTATION_EXPERIMENT = REPO_ROOT / "work_dirs/segmentation/coco2017_maskrcnn_dinov3_official_adapter"


def parse_args():
    parser = argparse.ArgumentParser(description="Run custom evaluation for DINOv3 experiments")
    parser.add_argument("--task", choices=["detection", "segmentation", "both"], default="both")
    parser.add_argument("--detection-exp", default=DEFAULT_DETECTION_EXPERIMENT)
    parser.add_argument("--segmentation-exp", default=DEFAULT_SEGMENTATION_EXPERIMENT)
    parser.add_argument("--detection-config", default=None)
    parser.add_argument("--segmentation-config", default=None)
    parser.add_argument("--checkpoint-select", choices=["all", "latest", "best", "final"], default="all")
    parser.add_argument("--low-mem", action="store_true", help="Enable low-memory runtime options")
    parser.add_argument("--samples-per-gpu", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument(
        "--extra-eval-option",
        action="append",
        default=[],
        help="Forwarded to eval_automation.py as key=value; repeatable",
    )
    return parser.parse_args()


def run_single(
    task,
    experiment_dir,
    config_path,
    checkpoint_select,
    extra_eval_option,
    low_mem,
    samples_per_gpu,
    workers_per_gpu,
    cuda_visible_devices,
    python_executable,
):
    command = [
        sys.executable,
        "detection/eval_automation.py",
        "--task",
        task,
        "--experiment-dir",
        str(Path(experiment_dir)),
        "--checkpoint-select",
        checkpoint_select,
    ]
    if config_path:
        command.extend(["--config", str(Path(config_path))])
    if low_mem:
        command.append("--low-mem")
        command.extend(["--samples-per-gpu", str(int(samples_per_gpu))])
        command.extend(["--workers-per-gpu", str(int(workers_per_gpu))])
    if cuda_visible_devices is not None:
        command.extend(["--cuda-visible-devices", str(cuda_visible_devices)])
    if python_executable is not None:
        command.extend(["--python-executable", str(python_executable)])
    for option in extra_eval_option:
        command.extend(["--extra-eval-option", option])

    subprocess.run(command, check=True)


def main():
    args = parse_args()

    if args.task in ("detection", "both"):
        run_single(
            task="detection",
            experiment_dir=args.detection_exp,
            config_path=args.detection_config,
            checkpoint_select=args.checkpoint_select,
            extra_eval_option=args.extra_eval_option,
            low_mem=args.low_mem,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            cuda_visible_devices=args.cuda_visible_devices,
            python_executable=args.python_executable,
        )

    if args.task in ("segmentation", "both"):
        run_single(
            task="segmentation",
            experiment_dir=args.segmentation_exp,
            config_path=args.segmentation_config,
            checkpoint_select=args.checkpoint_select,
            extra_eval_option=args.extra_eval_option,
            low_mem=args.low_mem,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            cuda_visible_devices=args.cuda_visible_devices,
            python_executable=args.python_executable,
        )


if __name__ == "__main__":
    main()
