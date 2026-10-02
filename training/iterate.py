"""自动执行自我对弈、回放训练、导出、验证和教师模型晋升。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .checkpoint import load_checkpoint


STATE_FORMAT = "xiangqi-nnue-iteration-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class IterationState:
    """可原子保存并在任务中断后恢复的迭代状态。

    属性:
        completed_generation: 最近完成并晋升的代数；``-1`` 表示尚无候选模型。
        champion_checkpoint: 当前教师 PyTorch 检查点路径，可以为空。
        champion_model: 当前教师 C++ NNUE 权重路径，可以为空。
        champion_epoch: 当前检查点已经训练到的总 epoch。
    """

    completed_generation: int = -1
    champion_checkpoint: str | None = None
    champion_model: str | None = None
    champion_epoch: int = 0


def run_command(command: Sequence[str]) -> None:
    """在项目根目录执行一个子命令并实时继承标准输入输出。

    参数:
        command: 已经拆分好的命令及参数，不经过 Shell 二次解析。

    异常:
        subprocess.CalledProcessError: 子命令返回非零状态时抛出。
    """

    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件，再原子替换迭代状态文件。

    参数:
        path: 状态 JSON 的最终路径。
        payload: 可以被标准 JSON 编码的状态字典。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    """原子更新便于其他程序引用的稳定冠军文件。

    参数:
        source: 已完整写好的源文件。
        destination: 要更新的 ``champion.pt`` 或 ``champion.nnue``。
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def load_state(path: Path) -> IterationState:
    """读取并校验已有迭代状态；文件不存在时返回初始状态。

    参数:
        path: ``state.json`` 路径。

    返回:
        恢复出的 ``IterationState``。

    异常:
        ValueError: 状态格式版本或字段非法时抛出。
    """

    if not path.exists():
        return IterationState()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("format") != STATE_FORMAT:
        raise ValueError(f"unsupported iteration state: {path}")
    state = IterationState(
        completed_generation=int(payload["completed_generation"]),
        champion_checkpoint=payload.get("champion_checkpoint"),
        champion_model=payload.get("champion_model"),
        champion_epoch=int(payload.get("champion_epoch", 0)),
    )
    if state.completed_generation < -1 or state.champion_epoch < 0:
        raise ValueError(f"invalid values in iteration state: {path}")
    return state


def checkpoint_epoch(path: Path | None) -> int:
    """返回检查点的已完成 epoch；空路径返回零。

    参数:
        path: 本项目格式的 PyTorch 检查点。
    """

    if path is None:
        return 0
    return int(load_checkpoint(path, "cpu").get("epoch", 0))


def replay_datasets(work_dir: Path, generation: int, count: int) -> list[Path]:
    """选择当前代及最近若干代已经生成的数据文件。

    参数:
        work_dir: 所有代次目录的共同根目录。
        generation: 当前正在训练的代数。
        count: 最多回放多少代，必须为正数。

    返回:
        按代数从旧到新排列的非空 JSONL 路径。
    """

    if count <= 0:
        raise ValueError("replay generation count must be positive")
    begin = max(0, generation - count + 1)
    paths = [
        work_dir / f"generation-{index:03d}" / "data.jsonl"
        for index in range(begin, generation + 1)
    ]
    result = [path for path in paths if path.is_file() and path.stat().st_size > 0]
    if not result:
        raise RuntimeError("no replay datasets are available")
    return result


def resolve_saved_path(value: str | None) -> Path | None:
    """把状态中的可选路径恢复为绝对 ``Path``。

    参数:
        value: 状态文件保存的路径字符串或空值。
    """

    return Path(value).resolve() if value else None


def generate_data(
    args: argparse.Namespace,
    generation: int,
    destination: Path,
    teacher_model: Path | None,
) -> None:
    """为一代生成完整的自我对弈 JSONL，并以重命名方式发布。

    参数:
        args: 总控脚本命令行配置。
        generation: 当前代数，用于派生可复现随机种子。
        destination: 该代最终 ``data.jsonl`` 路径。
        teacher_model: 上一代 NNUE；为空时使用手工评估教师。
    """

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"reuse completed dataset: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(args.generator),
        "selfplay",
        "--output",
        str(temporary),
        "--games",
        str(args.games_per_generation),
        "--play-depth",
        str(args.play_depth),
        "--label-depth",
        str(args.label_depth),
        "--qdepth",
        str(args.quiescence_depth),
        "--max-plies",
        str(args.max_plies),
        "--sample-start",
        str(args.sample_start),
        "--sample-min-gap",
        str(args.sample_min_gap),
        "--sample-max-gap",
        str(args.sample_max_gap),
        "--random-plies",
        str(args.random_plies),
        "--random-top-k",
        str(args.random_top_k),
        "--random-margin",
        str(args.random_margin),
        "--tt-mb",
        str(args.tt_mb),
        "--seed",
        str(args.seed + generation),
    ]
    if teacher_model is not None:
        command.extend(["--nnue", str(teacher_model)])
    run_command(command)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("data generator did not produce a non-empty dataset")
    os.replace(temporary, destination)


def train_candidate(
    args: argparse.Namespace,
    generation: int,
    datasets: Sequence[Path],
    previous_checkpoint: Path | None,
    output_dir: Path,
) -> Path:
    """从冠军或中断检查点训练当前代候选模型。

    参数:
        args: 总控脚本训练配置。
        generation: 当前代数，用于训练随机种子。
        datasets: 最近若干代的回放数据。
        previous_checkpoint: 上一代冠军检查点；第一代可以为空。
        output_dir: 当前代检查点目录。

    返回:
        当前代验证损失最好的检查点路径。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pt"
    best = output_dir / "best.pt"
    base_epoch = checkpoint_epoch(previous_checkpoint)
    target_epoch = base_epoch + args.epochs_per_generation

    if latest.is_file():
        resume = latest
        reset_best = False
    else:
        resume = previous_checkpoint
        reset_best = previous_checkpoint is not None

    current_epoch = checkpoint_epoch(resume)
    if current_epoch < target_epoch:
        command = [
            sys.executable,
            "-m",
            "training.train",
            *[str(path) for path in datasets],
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(target_epoch),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--validation-fraction",
            str(args.validation_fraction),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed + generation),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        if reset_best:
            command.append("--reset-best")
        run_command(command)
    else:
        print(
            f"reuse completed training: epoch {current_epoch} >= {target_epoch}",
            flush=True,
        )

    if not best.is_file():
        if latest.is_file():
            return latest
        raise RuntimeError("training produced neither best.pt nor latest.pt")
    return best


def parse_args() -> argparse.Namespace:
    """解析自动迭代、数据生成、训练和恢复参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("training/runs/iterations"))
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="number of additional generations to complete in this invocation",
    )
    parser.add_argument("--games-per-generation", type=int, default=20)
    parser.add_argument("--play-depth", type=int, default=2)
    parser.add_argument("--label-depth", type=int, default=4)
    parser.add_argument("--quiescence-depth", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--sample-start", type=int, default=8)
    parser.add_argument("--sample-min-gap", type=int, default=2)
    parser.add_argument("--sample-max-gap", type=int, default=4)
    parser.add_argument("--random-plies", type=int, default=10)
    parser.add_argument("--random-top-k", type=int, default=3)
    parser.add_argument("--random-margin", type=int, default=150)
    parser.add_argument("--tt-mb", type=int, default=16)
    parser.add_argument("--replay-generations", type=int, default=3)
    parser.add_argument("--epochs-per-generation", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--generator", type=Path, default=Path("build/make/xiangqi_generate_data"))
    parser.add_argument("--engine", type=Path, default=Path("build/make/xiangqi_cli"))
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-model", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在启动昂贵子任务前拒绝明显非法的迭代配置。"""

    positive_names = (
        "generations",
        "games_per_generation",
        "play_depth",
        "label_depth",
        "max_plies",
        "sample_min_gap",
        "sample_max_gap",
        "random_top_k",
        "tt_mb",
        "replay_generations",
        "epochs_per_generation",
        "batch_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.quiescence_depth < 0 or args.sample_start < 0 or args.random_plies < 0:
        raise ValueError("depth and ply offsets cannot be negative")
    if args.sample_min_gap > args.sample_max_gap:
        raise ValueError("minimum sample gap cannot exceed maximum sample gap")
    if args.random_margin < 0:
        raise ValueError("random margin cannot be negative")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")


def main() -> None:
    """连续完成指定数量的代次，并在每代成功后原子更新冠军。"""

    args = parse_args()
    validate_args(args)
    args.work_dir = args.work_dir.resolve()
    args.generator = (PROJECT_ROOT / args.generator).resolve() if not args.generator.is_absolute() else args.generator
    args.engine = (PROJECT_ROOT / args.engine).resolve() if not args.engine.is_absolute() else args.engine
    state_path = args.work_dir / "state.json"
    state = load_state(state_path)

    if state.completed_generation >= 0 and (
        args.bootstrap_checkpoint is not None or args.bootstrap_model is not None
    ):
        raise ValueError("bootstrap options cannot be changed after iteration has started")

    if not args.skip_build:
        run_command(["make", "-j2", "all"])
    if not args.generator.is_file() or not args.engine.is_file():
        raise FileNotFoundError("generator or C++ engine is missing; build the project first")

    champion_checkpoint = resolve_saved_path(state.champion_checkpoint)
    champion_model = resolve_saved_path(state.champion_model)
    if state.completed_generation < 0:
        champion_checkpoint = (
            args.bootstrap_checkpoint.resolve() if args.bootstrap_checkpoint else None
        )
        champion_model = args.bootstrap_model.resolve() if args.bootstrap_model else None
        if champion_checkpoint and not champion_checkpoint.is_file():
            raise FileNotFoundError(champion_checkpoint)
        if champion_model and not champion_model.is_file():
            raise FileNotFoundError(champion_model)
        if champion_checkpoint and champion_model is None:
            champion_model = args.work_dir / "bootstrap.nnue"
            run_command(
                [
                    sys.executable,
                    "-m",
                    "training.export_nnue",
                    str(champion_checkpoint),
                    str(champion_model),
                ]
            )

    print(
        "warning: automatic promotion verifies file integrity and Python/C++ parity, "
        "but does not yet prove that the candidate is stronger by arena matches",
        flush=True,
    )

    first_generation = state.completed_generation + 1
    final_generation = first_generation + args.generations
    for generation in range(first_generation, final_generation):
        print(f"\n===== generation {generation:03d} =====", flush=True)
        generation_dir = args.work_dir / f"generation-{generation:03d}"
        data_path = generation_dir / "data.jsonl"
        generate_data(args, generation, data_path, champion_model)
        datasets = replay_datasets(
            args.work_dir, generation, args.replay_generations
        )
        print(
            "replay datasets: " + ", ".join(str(path) for path in datasets),
            flush=True,
        )

        candidate_checkpoint = train_candidate(
            args,
            generation,
            datasets,
            champion_checkpoint,
            generation_dir / "checkpoints",
        )
        candidate_model = generation_dir / "candidate.nnue"
        run_command(
            [
                sys.executable,
                "-m",
                "training.export_nnue",
                str(candidate_checkpoint),
                str(candidate_model),
            ]
        )
        run_command(
            [
                sys.executable,
                "-m",
                "training.verify_cpp",
                str(candidate_checkpoint),
                "--engine",
                str(args.engine),
            ]
        )

        # 状态始终指向不会被后续代覆盖的文件，避免更新稳定别名时断电造成
        # “新检查点 + 旧模型”的混合教师。champion.* 只是方便外部程序使用的别名。
        champion_checkpoint = candidate_checkpoint.resolve()
        champion_model = candidate_model.resolve()
        state = IterationState(
            completed_generation=generation,
            champion_checkpoint=str(champion_checkpoint),
            champion_model=str(champion_model),
            champion_epoch=checkpoint_epoch(champion_checkpoint),
        )
        atomic_write_json(
            state_path,
            {"format": STATE_FORMAT, **asdict(state)},
        )
        stable_checkpoint = args.work_dir / "champion.pt"
        stable_model = args.work_dir / "champion.nnue"
        atomic_copy(champion_checkpoint, stable_checkpoint)
        atomic_copy(champion_model, stable_model)
        print(
            f"promoted generation {generation:03d}: {stable_model}",
            flush=True,
        )


if __name__ == "__main__":
    main()
