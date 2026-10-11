"""串联 Pikafish 标注、独立教师测试集、NNUE 训练与导出；不晋升冠军。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from .dataset import JsonlPositionDataset
from .diagnostics import write_json
from .partition import (
    DEFAULT_TEST_MANIFEST, canonical_position_key, file_fingerprint,
    load_test_manifest, source_id, write_test_manifest,
)
from .run_lock import run_lock

ROOT = Path(__file__).resolve().parents[1]


def run_stage(command: list[str], name: str, directory: Path, state: dict[str, Any]) -> None:
    """串行执行阶段并保存日志；出错保留状态，不继续训练不完整数据。"""

    if state["stages"].get(name, {}).get("complete"):
        return
    print(f"\n[{name}] $ {shlex.join(command)}", flush=True)
    state["stages"][name] = {"command": command, "complete": False}
    write_json(directory / "pipeline-state.json", state)
    with (directory / f"{name}.log").open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    state["stages"][name].update(returncode=code, complete=code == 0)
    write_json(directory / "pipeline-state.json", state)
    if code != 0:
        raise RuntimeError(f"{name} failed; inspect {directory / (name + '.log')}")


def prepare_teacher_test(directory: Path, protected_manifest: Path) -> Path:
    """教师测试集只能来自旧固定测试局；审计开发集没有整局或镜像泄漏。"""

    old = load_test_manifest(protected_manifest)
    old_groups, old_positions = set(old["groups"]), set(old["position_keys"])
    development, teacher_test = directory / "development.jsonl", directory / "fixed-test.jsonl"
    for path, is_test in ((development, False), (teacher_test, True)):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                group = record["original"]["game_group"]
                key = canonical_position_key(record["fen"])
                if is_test and (group not in old_groups or key not in old_positions):
                    raise ValueError("teacher test contains a non-held-out position")
                if not is_test and (group in old_groups or key in old_positions):
                    raise ValueError("development data leaks the old fixed test")
    samples = JsonlPositionDataset(teacher_test).samples
    payload = {
        "format": "xiangqi-fixed-test-v1", "seed": old["seed"], "fraction": old["fraction"],
        "groups": sorted({sample.group_key for sample in samples}),
        "position_keys": sorted({sample.position_key for sample in samples}),
        "source_fingerprints": {source_id(teacher_test): file_fingerprint(teacher_test)},
    }
    path = directory / "teacher-test-manifest.json"
    if path.exists():
        existing = load_test_manifest(path)
        if any(existing[key] != value for key, value in payload.items()):
            raise ValueError("teacher test identity changed")
    else:
        write_test_manifest(path, payload)
    return path


def main() -> None:
    """标注和训练可重入；配置变化需新目录，原数据和冠军不受影响。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--limit", type=int, default=10000, help="development positions; 0 means all")
    parser.add_argument("--test-limit", type=int, default=1000, help="protected test positions; 0 means all")
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpp-engine", type=Path, default=ROOT / "build/make/xiangqi_cli")
    args = parser.parse_args()
    if args.epochs <= 0 or args.nodes <= 0 or args.limit < 0 or args.test_limit < 0:
        parser.error("epochs/nodes must be positive and sample caps nonnegative")
    directory = args.work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "format": "xiangqi-pikafish-pipeline-v1",
        "args": {key: [str(p.resolve()) for p in value] if key == "datasets"
                 else str(value.resolve()) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "inputs": {source_id(p): file_fingerprint(p) for p in args.datasets},
        "engine_sha256": file_fingerprint(args.engine), "network_sha256": file_fingerprint(args.network),
        "cpp_engine_sha256": file_fingerprint(args.cpp_engine),
        "protected_manifest_sha256": file_fingerprint(args.test_manifest),
        "initialization": "random", "outcome_weight": 0.0,
        "score_scale": 600.0, "champion_promotion": False,
        "note": "teacher-test errors are in Pikafish cp units, not directly comparable to old handcrafted test MAE",
    }
    with run_lock(directory / "pipeline.lock"):
        plan_path, state_path = directory / "pipeline-plan.json", directory / "pipeline-state.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("pipeline inputs/config changed; use a new work directory")
        else:
            if state_path.exists() or (directory / "model").exists():
                raise FileExistsError("existing training run has no matching pipeline plan")
            write_json(plan_path, plan)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "running", "stages": {}}
        execute_pipeline(args, directory, state)


def execute_pipeline(args: argparse.Namespace, directory: Path, state: dict[str, Any]) -> None:
    """在独占锁内执行；保留训练总预算，完成后检查模型身份再返回。"""

    state_path = directory / "pipeline-state.json"
    if state["status"] == "trained_not_promoted":
        if file_fingerprint(Path(state["model"])) != state["model_sha256"]:
            raise ValueError("completed candidate changed after training")
        print(f"already trained; champion unchanged: {directory / 'model/best.nnue'}")
        return
    state["status"] = "running"
    common = [sys.executable, "-u", "-m", "training.relabel", *map(str, args.datasets),
              "--engine", str(args.engine), "--network", str(args.network),
              "--test-manifest", str(args.test_manifest), "--nodes", str(args.nodes), "--seed", str(args.seed)]
    try:
        for name, limit, partition in (("development", args.limit, "development"),
                                        ("fixed-test", args.test_limit, "fixed-test")):
            output = directory / f"{name}.jsonl"
            # 标注器自己校验精确配置与数据库，避免仅凭 stage 状态相信旧数据。
            state["stages"].pop(f"label-{name}", None)
            run_stage([*common, "--output", str(output), "--limit", str(limit), "--partition", partition],
                      f"label-{name}", directory, state)
        test_manifest = prepare_teacher_test(directory, args.test_manifest)
        model = directory / "model"
        train_command = [sys.executable, "-u", "-m", "training.train", str(directory / "development.jsonl"),
                         "--output-dir", str(model), "--test-manifest", str(test_manifest),
                         "--epochs", str(args.epochs), "--batch-size", "512", "--learning-rate", "0.001",
                         "--score-scale", "600", "--outcome-weight", "0", "--seed", str(args.seed),
                         "--split-seed", str(args.seed), "--device", args.device]
        if not state["stages"].get("train", {}).get("complete") and (model / "latest.pt").exists():
            from .checkpoint import load_checkpoint
            finished = int(load_checkpoint(model / "latest.pt", "cpu")["epoch"])
            # 完成最后一轮但报告未落盘时，用只收尾模式，不额外增加训练轮次。
            train_command[train_command.index("--epochs") + 1] = str(args.epochs if finished < args.epochs else 0)
            train_command.extend(["--resume", str(model / "latest.pt")])
        run_stage(train_command, "train", directory, state)
        run_stage([sys.executable, "-m", "training.export_nnue", str(model / "best.pt"), str(model / "best.nnue")],
                  "export", directory, state)
        run_stage([sys.executable, "-m", "training.verify_cpp", str(model / "best.pt"),
                   "--engine", str(args.cpp_engine), "--output-json", str(model / "parity.json")],
                  "parity", directory, state)
        state.update(status="trained_not_promoted", model=str(model / "best.nnue"),
                     model_sha256=file_fingerprint(model / "best.nnue"), champion_changed=False)
        print(f"trained candidate (not promoted): {model / 'best.nnue'}", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(state_path, state)


if __name__ == "__main__":
    main()
