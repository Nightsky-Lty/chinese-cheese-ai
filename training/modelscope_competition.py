"""扩充 nowcan 数据、训练可配置架构候选，并通过完整门禁竞争冠军。

所有阶段有独立日志和状态；自动晋升须通过门禁，不依赖已清理的历史实验目录。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from .arena import (
    DEFAULT_MAX_UNFINISHED, DEFAULT_OPENINGS, DEFAULT_REQUIRED_SCORE,
    DEFAULT_REQUIRED_WIN_IMPROVEMENT,
    PROMOTION_POLICY, Opponent, run_arena,
)
from .baseline import validation_gate
from .checkpoint import load_checkpoint
from .diagnostics import write_json
from .iterate import atomic_copy, next_arena_report_path
from .model import NetworkConfig
from .partition import file_fingerprint, load_test_manifest
from .pikafish_pipeline import run_stage
from .run_lock import run_lock

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/modelscope-nowcan/processed"


def now() -> str:
    """为运行记录提供带时区的 UTC 时间，不用于决定训练或晋升。"""

    return datetime.now(timezone.utc).isoformat()


def artifact(path: Path) -> dict[str, str]:
    """绑定文件的绝对路径和 SHA-256；输入不存在时立即报错。"""

    return {"path": str(path.resolve()), "sha256": file_fingerprint(path)}


def verify_artifact(record: dict[str, str]) -> Path:
    """拒绝恢复期间被修改的输入、冠军或候选产物。"""

    path = Path(record["path"])
    if file_fingerprint(path) != record["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def incumbent(store: Path) -> dict[str, Any]:
    """读取共同目录中的冠军；无冠军登记时要求显式建立，不再依赖旧实验。"""

    with run_lock(store / "champion.lock"):
        state_path = store / "state.json"
        if state_path.exists():
            record = json.loads(state_path.read_text())
            if record.get("format") != "xiangqi-champion-registry-v1":
                raise ValueError("unsupported champion registry")
            verify_artifact(record["model"])
            verify_artifact(record["checkpoint"])
            return record
        raise FileNotFoundError(f"champion registry missing: {state_path}; select an initial champion explicitly")


def execute_stage(command: list[str], name: str, directory: Path, state: dict[str, Any]) -> None:
    """记录阶段开始/结束时间、退出码和日志；同配置完成阶段不重跑。"""

    if state["stages"].get(name, {}).get("complete"):
        return
    state["status"] = name
    started = time.monotonic()
    timestamp = now()
    try:
        run_stage(command, name, directory, state)
    finally:
        stage = state["stages"].setdefault(name, {})
        stage.update(started_at=timestamp, finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(directory / "competition-state.json", state)


def run(args: argparse.Namespace) -> None:
    """百万局面实验：固定验证、独立测试、可配置架构训练和冠军门禁。"""

    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset_dir.resolve()
    store = args.champion_store.resolve()
    plan_path = directory / "competition-plan.json"
    state_path = directory / "competition-state.json"
    # Closed experiments remain readable after obsolete opponents have been removed.
    if state_path.exists():
        completed = json.loads(state_path.read_text())
        if completed.get("status") in {"promoted", "rejected", "stale_incumbent"}:
            for record in completed.get("candidate", {}).values():
                verify_artifact(record)
            print(f"already complete: {completed['status']}; report: {state_path}", flush=True)
            return
    network = NetworkConfig(args.accumulator_size, tuple(args.hidden_sizes))
    config = {"architecture": network.name, "train_count": args.train_count,
              "validation_count": 10000, "test_count": 10000, "epochs": args.epochs,
              "batch_size": args.batch_size, "device": args.device, "seed": args.seed,
              "initialization": "random", "score_scale": 600, "outcome_weight": 0,
              "dataset_dir": str(dataset), "champion_store": str(store),
              "fixed_validation": artifact(DATA / "baseline-100k-v1/validation.jsonl"),
              "audit": artifact(DATA / "score-audit.json"),
              "protected_test": artifact(args.protected_test),
              "history": artifact(ROOT / "training/runs/modelscope-nowcan-100k-v1/best.nnue"),
              "cli_engine": artifact(ROOT / "build/make/xiangqi_cli"),
              "protocol_engine": artifact(ROOT / "build/make/xiangqi_protocol"),
              "openings": artifact(args.openings),
              "arena": {"mode": "time", "depth_cap": 64, "time_limit_ms": args.time_limit_ms,
                        "promotion_policy": PROMOTION_POLICY,
                        "min_games": 100, "min_independent_pairs": 50, "required_score": args.required_score,
                        "required_win_improvement": args.required_win_improvement,
                        "max_unfinished": args.max_unfinished, "max_plies": args.max_plies}}
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["config"] != config:
            raise ValueError("competition inputs/config changed; use a new work directory")
        current = plan["incumbent"]
        verify_artifact(current["model"])
        verify_artifact(current["checkpoint"])
        state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
        if state.get("status") in {"promoted", "rejected", "stale_incumbent"}:
            print(f"already complete: {state['status']}; report: {state_path}", flush=True)
            return
    else:
        if (directory / "best.pt").exists() or (directory / "latest.pt").exists():
            raise FileExistsError("checkpoint exists without a competition plan")
        current = incumbent(store)
        write_json(plan_path, {"format": "nowcan-competition-plan-v1", "created_at": now(),
                               "config": config, "incumbent": current})
        state = {"format": "nowcan-competition-state-v1", "status": "starting", "stages": {},
                 "champion_changed": False, "started_at": now()}
        write_json(state_path, state)

    # An already-running sampler can finish first; if it stops unsuccessfully,
    # our prepare stage resumes its per-game transactions instead of waiting forever.
    announced = False
    while True:
        try:
            with run_lock(dataset / "sampling.lock"):
                pass
            break
        except RuntimeError:
            if not announced:
                state["status"] = "waiting_for_sampling"
                write_json(state_path, state)
                print("waiting for active sampler; training and arena will start automatically", flush=True)
                announced = True
            time.sleep(5)
    prepare = [sys.executable, "-u", "-m", "training.modelscope_data", "prepare",
               "--audit", config["audit"]["path"], "--convention", "post-mover",
               "--train-count", str(args.train_count), "--validation-count", "10000",
               "--fixed-validation", config["fixed_validation"]["path"], "--test-count", "10000",
               "--test-manifest", config["protected_test"]["path"],
               "--seed", str(args.seed), "--output", str(dataset)]
    execute_stage(prepare, "prepare", directory, state)
    sample_report = json.loads((dataset / "sampling-report.json").read_text())
    for filename, field in (("train.jsonl", "train_sha256"), ("validation.jsonl", "validation_sha256"),
                            ("test.jsonl", "test_sha256"), ("test-manifest.json", "test_manifest_sha256")):
        if file_fingerprint(dataset / filename) != sample_report[field]:
            raise ValueError(f"prepared sample changed: {filename}")
    if sample_report["train_count"] != args.train_count:
        raise ValueError("unexpected training count")
    load_test_manifest(dataset / "test-manifest.json")
    sampled_identity = artifact(dataset / "sampling-report.json")
    if "sampled_data" in state and state["sampled_data"] != sampled_identity:
        raise ValueError("sampling report changed during competition")
    state["sampled_data"] = sampled_identity
    write_json(state_path, state)

    train = [sys.executable, "-u", "-m", "training.train", str(dataset / "train.jsonl"),
             "--validation-dataset", str(dataset / "validation.jsonl"),
             "--test-manifest", str(dataset / "test-manifest.json"), "--output-dir", str(directory),
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--device", args.device,
             "--score-scale", "600", "--outcome-weight", "0", "--seed", str(args.seed)]
    train.extend(["--accumulator-size", str(network.accumulator_size),
                  "--hidden-sizes", *map(str, network.hidden_sizes)])
    latest = directory / "latest.pt"
    if latest.exists() and not state["stages"].get("train", {}).get("complete"):
        train.extend(["--resume", str(latest)])
        if int(load_checkpoint(latest, "cpu")["epoch"]) >= args.epochs:
            train[train.index("--epochs") + 1] = "0"
    execute_stage(train, "train", directory, state)
    validation = validation_gate(directory / "report.json", directory / "best.pt")
    state["validation"] = validation
    if not validation["passed"]:
        raise RuntimeError(f"candidate validation gate failed: {validation['reason']}")

    for name, command in (
        ("export", [sys.executable, "-m", "training.export_nnue", str(directory / "best.pt"), str(directory / "best.nnue")]),
        ("parity", [sys.executable, "-m", "training.verify_cpp", str(directory / "best.pt"),
                    "--engine", config["cli_engine"]["path"], "--output-json", str(directory / "parity.json")]),
    ):
        execute_stage(command, name, directory, state)
    candidate = {"checkpoint": artifact(directory / "best.pt"), "model": artifact(directory / "best.nnue")}
    if "candidate" in state and state["candidate"] != candidate:
        raise ValueError("candidate artifact changed after verification")
    state["candidate"] = candidate
    write_json(state_path, state)
    if not state["stages"].get("arena", {}).get("complete"):
        state["status"] = "arena"
        arena_path = next_arena_report_path(directory)
        state["stages"]["arena"] = {"complete": False, "report": str(arena_path), "started_at": now()}
        write_json(state_path, state)
        started = time.monotonic()
        opponents = [Opponent("hand", None), Opponent("champion", Path(current["model"]["path"])),
                     Opponent("history-modelscope-100k", Path(config["history"]["path"]))]
        arena = run_arena(Path(config["protocol_engine"]["path"]), directory / "best.nnue", opponents,
                          arena_path, search_mode="time", depth=64, time_limit_ms=args.time_limit_ms,
                          max_plies=args.max_plies, min_games=100, min_independent_pairs=50,
                          required_score=args.required_score, required_win_improvement=args.required_win_improvement,
                          max_unfinished=args.max_unfinished, openings_path=args.openings)
        state["stages"]["arena"].update(complete=True, passed=arena["passed"],
                                         elapsed_seconds=time.monotonic() - started, finished_at=now())
        write_json(state_path, state)
    else:
        arena = json.loads(Path(state["stages"]["arena"]["report"]).read_text())
    if not arena["passed"]:
        state.update(status="rejected", finished_at=now(), champion_changed=False,
                     reason="arena promotion target or protocol safety gate failed")
        write_json(state_path, state)
        print(f"candidate retained; champion unchanged; report: {state_path}", flush=True)
        return

    # Hold the registry lock only while committing. A changed incumbent requires
    # a fresh competition, never a silent overwrite of a newer champion.
    with run_lock(store / "champion.lock"):
        live = json.loads((store / "state.json").read_text())
        if live["model"] != current["model"]:
            state.update(status="stale_incumbent", finished_at=now(), champion_changed=False)
            write_json(state_path, state)
            return
        for item in (candidate["checkpoint"], candidate["model"], current["model"], config["history"],
                     config["protocol_engine"], config["cli_engine"], config["openings"]):
            verify_artifact(item)
        validation = validation_gate(directory / "report.json", directory / "best.pt")
        if not validation["passed"] or not state["stages"]["parity"]["complete"]:
            raise RuntimeError("promotion evidence changed")
        updated = {"format": "xiangqi-champion-registry-v1", "updated_at": now(),
                   "origin": "modelscope-competition", **candidate, "previous": current,
                   "competition_report": str(state_path), "arena_report": artifact(Path(state["stages"]["arena"]["report"]))}
        atomic_copy(directory / "best.pt", store / "champion.pt")
        atomic_copy(directory / "best.nnue", store / "champion.nnue")
        write_json(store / "state.json", updated)
    state.update(status="promoted", finished_at=now(), champion_changed=True,
                 promoted_model=str(store / "champion.nnue"))
    write_json(state_path, state)
    print(f"promoted: {store / 'champion.nnue'}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析固定样本规模、训练设置和等时间竞技场预算。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATA / "competition-1m-v1")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "training/runs/modelscope-nowcan-1m-v1")
    parser.add_argument("--champion-store", type=Path, default=ROOT / "training/runs/champions")
    parser.add_argument("--train-count", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--accumulator-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--time-limit-ms", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--required-score", type=float, default=DEFAULT_REQUIRED_SCORE,
                        help="baseline paired score without a champion (default: 0.55)")
    parser.add_argument("--required-win-improvement", type=float, default=DEFAULT_REQUIRED_WIN_IMPROVEMENT,
                        help="minimum same-color win-rate difference for BOTH colors vs champion (0.15 = 15 percentage points)")
    parser.add_argument("--max-unfinished", type=int, default=DEFAULT_MAX_UNFINISHED,
                        help="ply-cap allowance (default: 10), also capped at 10%%")
    parser.add_argument("--protected-test", type=Path,
                        default=DATA / "competition-1m-v1/test-manifest.json",
                        help="fixed test anchors excluded from newly sampled datasets")
    args = parser.parse_args()
    if min(args.train_count, args.epochs, args.batch_size, args.time_limit_ms, args.max_plies) <= 0 or args.max_unfinished < 0:
        parser.error("sample, epoch, batch, time and ply limits must be positive")
    if not 0.0 <= args.required_score <= 1.0:
        parser.error("required score must be between 0 and 1")
    if not 0.0 <= args.required_win_improvement <= 1.0:
        parser.error("required win improvement must be between 0 and 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    with run_lock(arguments.work_dir.resolve() / "competition.lock"):
        try:
            run(arguments)
        except BaseException as error:
            path = arguments.work_dir.resolve() / "competition-state.json"
            if path.exists():
                payload = json.loads(path.read_text())
                payload.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                               error=f"{type(error).__name__}: {error}", failed_at=now())
                write_json(path, payload)
            raise
