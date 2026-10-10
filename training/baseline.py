"""从可信 JSONL 数据训练、验证、导出并按 arena 结果晋升首个 NNUE。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .arena import (
    DEFAULT_OPENINGS,
    Opponent,
    load_openings,
    resolve_search_depth,
    run_arena,
    sha256_file,
)
from .partition import DEFAULT_QUARANTINE_MANIFEST, DEFAULT_TEST_MANIFEST
from .iterate import candidate_validation_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_FORMAT = "xiangqi-nnue-baseline-v1"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_command(command: Sequence[str], label: str) -> dict[str, Any]:
    print(f"\n$ {shlex.join(command)}", flush=True)
    completed = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def validation_gate(
    report_path: Path,
    checkpoint_path: Path,
    max_validation_loss: float | None = None,
    max_fixed_test_score_loss: float | None = None,
) -> dict[str, Any]:
    """要求训练验证集与独立固定测试均实际运行且产出有限指标。"""

    if not report_path.is_file() or not checkpoint_path.is_file():
        return {"passed": False, "reason": "training report or best checkpoint missing"}
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("format") != "xiangqi-nnue-training-report-v1":
        return {"passed": False, "reason": "unsupported training report format"}
    checkpoint = torch_load_checkpoint(checkpoint_path)
    evidence = candidate_validation_evidence(report, checkpoint)
    split_counts = report.get("config", {}).get("split", {}).get("counts", {})
    fixed_test = report.get("fixed_test")
    best_loss = evidence.get("report_best_validation_loss")
    checkpoint_loss = evidence.get("checkpoint_validation_loss")
    best_loss = float(best_loss) if best_loss is not None else float("nan")
    checkpoint_loss = (
        float(checkpoint_loss) if checkpoint_loss is not None else float("nan")
    )
    fixed_loss = (
        float(fixed_test.get("score_loss", float("nan")))
        if isinstance(fixed_test, dict) else float("nan")
    )
    validation_count = int(split_counts.get("validation", 0))
    test_count = int(split_counts.get("test", 0))
    fixed_count = int(fixed_test.get("sample_count", 0)) if isinstance(fixed_test, dict) else 0
    passed = (
        evidence.get("passed") is True
        and math.isfinite(best_loss)
        and math.isfinite(checkpoint_loss)
        and math.isfinite(fixed_loss)
        and (max_validation_loss is None or best_loss <= max_validation_loss)
        and (
            max_fixed_test_score_loss is None
            or fixed_loss <= max_fixed_test_score_loss
        )
    )
    return {
        "passed": passed,
        "validation_samples": validation_count,
        "fixed_test_manifest_samples": test_count,
        "fixed_test_evaluated_samples": fixed_count,
        "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "fixed_test_score_loss": fixed_loss if math.isfinite(fixed_loss) else None,
        "max_validation_loss": max_validation_loss,
        "max_fixed_test_score_loss": max_fixed_test_score_loss,
        "identity": {
            "split_matches_checkpoint": evidence.get("split_matches_checkpoint", False),
            "best_epoch_matches_checkpoint": evidence.get("best_epoch_matches_checkpoint", False),
            "best_validation_loss_matches_checkpoint": evidence.get(
                "best_validation_loss_matches_checkpoint", False
            ),
        },
        "reason": None if passed else evidence.get(
            "reason", "validation or fixed-test threshold failed"
        ),
        "report": str(report_path.resolve()),
    }


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    # Keep loading centralized in the project helper, including its safe format checks.
    from .checkpoint import load_checkpoint

    return load_checkpoint(path, "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="new run directory; an existing directory is never overwritten")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-validation-loss", type=float)
    parser.add_argument("--max-fixed-test-score-loss", type=float)
    parser.add_argument("--score-scale", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--quarantine-manifest", type=Path, default=DEFAULT_QUARANTINE_MANIFEST)
    parser.add_argument("--engine", type=Path, default=Path("build/make/xiangqi_cli"))
    parser.add_argument("--protocol-engine", type=Path, default=Path("build/make/xiangqi_protocol"))
    parser.add_argument("--champion", type=Path,
                        help="optional current champion to include as a fixed opponent")
    parser.add_argument("--history", type=Path, action="append", default=[],
                        help="fixed historical opponent; may be repeated")
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--search-mode", choices=("depth", "time"), default="time")
    parser.add_argument(
        "--depth", type=int, default=None,
        help="search depth/cap (default: 64 in time mode, 4 in depth mode)",
    )
    parser.add_argument("--time-limit-ms", type=int, default=150)
    parser.add_argument("--protocol-timeout", type=float, default=10.0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--required-score", type=float, default=0.70)
    parser.add_argument("--max-unfinished", type=int, default=0)
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--min-independent-pairs", type=int, default=50)
    return parser.parse_args()


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    args.depth = resolve_search_depth(args.search_mode, args.depth)
    if args.work_dir.exists():
        raise FileExistsError(f"baseline run directory already exists: {args.work_dir}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.depth <= 0 or (args.search_mode == "time" and args.time_limit_ms <= 0):
        raise ValueError("arena depth must be positive and timed searches need positive movetime")
    if not 0.0 <= args.validation_fraction < 1.0 or args.score_scale <= 0.0:
        raise ValueError("invalid training validation fraction or score scale")
    if (
        args.max_validation_loss is not None
        and (not math.isfinite(args.max_validation_loss) or args.max_validation_loss < 0.0)
    ) or (
        args.max_fixed_test_score_loss is not None
        and (
            not math.isfinite(args.max_fixed_test_score_loss)
            or args.max_fixed_test_score_loss < 0.0
        )
    ):
        raise ValueError("configured validation loss thresholds cannot be negative")
    if args.min_games <= 0 or args.min_games % 2 or args.min_independent_pairs <= 0:
        raise ValueError("min-games must be positive and even; min-independent-pairs positive")
    if not 0.0 <= args.required_score <= 1.0 or args.max_unfinished < 0:
        raise ValueError("invalid arena promotion threshold")

    args.work_dir = args.work_dir.resolve()
    args.datasets = [path.resolve() for path in args.datasets]
    args.test_manifest = args.test_manifest.resolve()
    args.quarantine_manifest = args.quarantine_manifest.resolve()
    args.engine = args.engine.resolve() if args.engine.is_absolute() else (PROJECT_ROOT / args.engine).resolve()
    args.protocol_engine = (
        args.protocol_engine.resolve() if args.protocol_engine.is_absolute()
        else (PROJECT_ROOT / args.protocol_engine).resolve()
    )
    args.champion = args.champion.resolve() if args.champion else None
    args.history = [path.resolve() for path in args.history]
    args.openings = args.openings.resolve()
    for path in [*args.datasets, args.test_manifest, args.quarantine_manifest,
                 args.engine, args.protocol_engine, args.openings]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.champion and not args.champion.is_file():
        raise FileNotFoundError(args.champion)
    if any(not path.is_file() for path in args.history):
        raise FileNotFoundError("one or more historical opponent models are missing")
    load_openings(args.openings)

    args.work_dir.mkdir(parents=True)
    checkpoint_dir = args.work_dir / "checkpoints"
    candidate_checkpoint = checkpoint_dir / "best.pt"
    candidate_model = args.work_dir / "candidate.nnue"
    config = {
        "format": BASELINE_FORMAT,
        "datasets": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.datasets
        ],
        "test_manifest": {
            "path": str(args.test_manifest), "sha256": sha256_file(args.test_manifest),
        },
        "quarantine_manifest": {
            "path": str(args.quarantine_manifest),
            "sha256": sha256_file(args.quarantine_manifest),
        },
        "training": {
            "initialization": "random",
            "score_only": True,
            "outcome_weight": 0.0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "validation_fraction": args.validation_fraction,
            "max_validation_loss": args.max_validation_loss,
            "max_fixed_test_score_loss": args.max_fixed_test_score_loss,
            "score_scale": args.score_scale,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "device": args.device,
            "workers": args.workers,
        },
        "parity": {"engine": str(args.engine), "engine_sha256": sha256_file(args.engine)},
        "arena": {
            "search_mode": args.search_mode,
            "protocol_engine": str(args.protocol_engine),
            "protocol_engine_sha256": sha256_file(args.protocol_engine),
            "openings": str(args.openings),
            "openings_sha256": sha256_file(args.openings),
            "depth": args.depth,
            "time_limit_ms": args.time_limit_ms,
            "protocol_timeout": args.protocol_timeout,
            "max_plies": args.max_plies,
            "required_score": args.required_score,
            "max_unfinished": args.max_unfinished,
            "min_games": args.min_games,
            "min_independent_pairs": args.min_independent_pairs,
        },
        "opponents": [
            {"name": "hand", "network": None},
            *([{"name": "champion", "network": str(args.champion),
                 "sha256": sha256_file(args.champion)}] if args.champion else []),
            *[
                {"name": f"history-{index}", "network": str(path), "sha256": sha256_file(path)}
                for index, path in enumerate(args.history, start=1)
            ],
        ],
    }
    atomic_write_json(args.work_dir / "config.json", config)
    report: dict[str, Any] = {
        "format": BASELINE_FORMAT,
        "config": config,
        "passed": False,
        "validation": {"passed": False, "status": "not_run"},
        "parity": {"passed": False, "status": "not_run"},
        "arena": {"passed": False, "status": "not_run"},
        "promotion": {"passed": False, "status": "not_promoted"},
    }
    report_path = args.work_dir / "report.json"
    atomic_write_json(report_path, report)

    try:
        train_command = [
            sys.executable, "-m", "training.train",
            *[str(path) for path in args.datasets],
            "--output-dir", str(checkpoint_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--validation-fraction", str(args.validation_fraction),
            "--score-scale", str(args.score_scale),
            "--outcome-weight", "0",
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--split-seed", str(args.split_seed),
            "--test-manifest", str(args.test_manifest),
            "--quarantine-manifest", str(args.quarantine_manifest),
            "--device", args.device,
        ]
        training = run_command(train_command, "training")
        report["training"] = {
            "passed": training["passed"],
            "returncode": training["returncode"],
            "stdout": training["stdout"],
            "stderr": training["stderr"],
            "report": str((checkpoint_dir / "report.json").resolve()),
        }
        if not training["passed"]:
            raise RuntimeError("training failed; no candidate will be promoted")

        validation = validation_gate(
            checkpoint_dir / "report.json",
            candidate_checkpoint,
            args.max_validation_loss,
            args.max_fixed_test_score_loss,
        )
        report["validation"] = validation
        atomic_write_json(report_path, report)
        if not validation["passed"]:
            raise RuntimeError(validation["reason"])

        export_command = [
            sys.executable, "-m", "training.export_nnue",
            str(candidate_checkpoint), str(candidate_model),
        ]
        exported = run_command(export_command, "export")
        report["export"] = {"passed": exported["passed"], "returncode": exported["returncode"]}
        if not exported["passed"]:
            raise RuntimeError("NNUE export failed")

        parity = run_command(
            [
                sys.executable, "-m", "training.verify_cpp",
                str(candidate_checkpoint), "--engine", str(args.engine),
            ],
            "python_cpp_parity",
        )
        report["parity"] = {
            "passed": parity["passed"],
            "returncode": parity["returncode"],
            "stdout": parity["stdout"],
            "stderr": parity["stderr"],
        }
        atomic_write_json(report_path, report)
        if not parity["passed"]:
            raise RuntimeError("Python/C++ parity verification failed")

        opponents = [Opponent("hand", None)]
        if args.champion:
            opponents.append(Opponent("champion", args.champion))
        opponents.extend(
            Opponent(f"history-{index}", path)
            for index, path in enumerate(args.history, start=1)
        )
        arena = run_arena(
            args.protocol_engine,
            candidate_model,
            opponents,
            args.work_dir / "arena.json",
            search_mode=args.search_mode,
            depth=args.depth,
            time_limit_ms=args.time_limit_ms,
            max_plies=args.max_plies,
            required_score=args.required_score,
            max_unfinished=args.max_unfinished,
            min_games=args.min_games,
            min_independent_pairs=args.min_independent_pairs,
            openings_path=args.openings,
            timeout_seconds=args.protocol_timeout,
        )
        report["arena"] = {
            "passed": arena["passed"],
            "report": str((args.work_dir / "arena.json").resolve()),
            "opponents": {
                name: result["summary"] for name, result in arena["opponents"].items()
            },
        }
        atomic_write_json(report_path, report)
        if not arena["passed"]:
            raise RuntimeError("one or more arena promotion gates failed")

        # The promotion marker is written last, after both artifacts are copied.
        for source, destination in (
            (candidate_checkpoint, args.work_dir / "champion.pt"),
            (candidate_model, args.work_dir / "champion.nnue"),
        ):
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        report["promotion"] = {
            "passed": True,
            "status": "promoted",
            "checkpoint": str((args.work_dir / "champion.pt").resolve()),
            "checkpoint_sha256": sha256_file(args.work_dir / "champion.pt"),
            "model": str((args.work_dir / "champion.nnue").resolve()),
            "model_sha256": sha256_file(args.work_dir / "champion.nnue"),
        }
        report["passed"] = True
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_write_json(report_path, report)
    print(f"baseline report: {report_path}; promotion={report['promotion']['status']}", flush=True)
    return report


def main() -> None:
    run_baseline(parse_args())


if __name__ == "__main__":
    main()
