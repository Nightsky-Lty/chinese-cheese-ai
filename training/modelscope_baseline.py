"""用固定 nowcan 抽样训练原架构基线，记录日志、续训并导出，不晋升冠军。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .checkpoint import load_checkpoint
from .diagnostics import write_json
from .model import ARCHITECTURE_NAME
from .partition import file_fingerprint
from .pikafish_pipeline import run_stage
from .run_lock import run_lock

ROOT = Path(__file__).resolve().parents[1]


def main(args: argparse.Namespace) -> None:
    """训练原 HalfKA 结构；同目录重跑时严格验证样本和配置是否一致。"""

    dataset = args.dataset_dir.resolve()
    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    train, validation = dataset / "train.jsonl", dataset / "validation.jsonl"
    report = json.loads((dataset / "sampling-report.json").read_text())
    if report.get("status") != "complete":
        raise ValueError("sampling is not complete")
    if file_fingerprint(train) != report["train_sha256"] or file_fingerprint(validation) != report["validation_sha256"]:
        raise ValueError("sampled data changed after preparation")
    plan = {"format": "nowcan-baseline-plan-v1", "architecture": ARCHITECTURE_NAME,
            "train": str(train), "validation": str(validation),
            "sampling_report_sha256": file_fingerprint(dataset / "sampling-report.json"),
            "test_manifest": str(args.test_manifest.resolve()),
            "protected_manifest_sha256": file_fingerprint(args.test_manifest),
            "epochs": args.epochs, "batch_size": args.batch_size, "device": args.device,
            "seed": args.seed, "score_scale": 600.0, "outcome_weight": 0.0,
            "fixed_test_scoring": "skipped: old teacher scores have a different unit",
            "champion_promotion": False}
    plan_path, state_path = directory / "baseline-plan.json", directory / "baseline-state.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("baseline configuration changed; use a new work directory")
        state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    else:
        if (directory / "best.pt").exists() or (directory / "latest.pt").exists():
            raise FileExistsError("checkpoint exists without its baseline plan")
        write_json(plan_path, plan)
        state = {"format": "nowcan-baseline-state-v1", "status": "running", "stages": {}}
        write_json(state_path, state)
    # run_stage uses pipeline-state.json internally; keep the baseline state too.
    if not state["stages"].get("train", {}).get("complete"):
        command = [sys.executable, "-u", "-m", "training.train", str(train),
                   "--validation-dataset", str(validation), "--test-manifest", str(args.test_manifest.resolve()),
                   "--skip-fixed-test", "--output-dir", str(directory), "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size), "--device", args.device,
                   "--score-scale", "600", "--outcome-weight", "0", "--seed", str(args.seed)]
        latest = directory / "latest.pt"
        if latest.exists():
            epoch = int(load_checkpoint(latest, "cpu")["epoch"])
            command.extend(["--resume", str(latest)])
            if epoch >= args.epochs:
                command[command.index("--epochs") + 1] = "0"
        try:
            run_stage(command, "train", directory, state)
        finally:
            write_json(state_path, state)
    for stage, command in (
        ("export", [sys.executable, "-m", "training.export_nnue", str(directory / "best.pt"), str(directory / "best.nnue")]),
        ("parity", [sys.executable, "-m", "training.verify_cpp", str(directory / "best.pt"),
                    "--engine", str(args.cpp_engine.resolve()), "--output-json", str(directory / "parity.json")]),
    ):
        if not state["stages"].get(stage, {}).get("complete"):
            try:
                run_stage(command, stage, directory, state)
            finally:
                write_json(state_path, state)
    state.update(status="trained_not_promoted", model=str(directory / "best.nnue"),
                 model_sha256=file_fingerprint(directory / "best.nnue"), champion_changed=False)
    write_json(state_path, state)
    print(f"complete: {directory / 'best.nnue'}; architecture unchanged; champion unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "training/data/modelscope-nowcan/processed/baseline-100k-v1")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "training/runs/modelscope-nowcan-100k-v1")
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "training/runs/pikafish-full-20261011/teacher-test-manifest.json")
    parser.add_argument("--cpp-engine", type=Path, default=ROOT / "build/make/xiangqi_cli")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    with run_lock(arguments.work_dir.resolve() / "baseline.lock"):
        main(arguments)
