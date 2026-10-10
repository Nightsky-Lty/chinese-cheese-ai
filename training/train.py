"""使用局面评分和可选赛果训练 HalfKA NNUE，未知赛果只参与评分损失。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .checkpoint import (
    checkpoint_score_scale,
    load_checkpoint,
    migrate_output_score_scale,
    save_checkpoint,
)
from .dataset import JsonlPositionDataset, TrainingSample, collate_samples
from .diagnostics import EpochMetrics, append_jsonl, write_json
from .model import DEFAULT_SCORE_SCALE, HalfKANetwork, NnueBatch
from .partition import (
    DEFAULT_QUARANTINE_MANIFEST, DEFAULT_TEST_MANIFEST,
    file_fingerprint, load_quarantine_manifest, load_test_manifest,
    manifest_source_path, split_indices,
)


def choose_device(requested: str) -> torch.device:
    """选择训练设备。

    参数:
        requested: 明确的 PyTorch 设备名，或自动选择用的 ``auto``。

    返回:
        ``auto`` 按 CUDA、MPS、CPU 顺序选择，否则直接解析用户给出的设备名。
    """

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_huber_loss(
    predictions: torch.Tensor, batch: NnueBatch, beta: float,
    score_scale: float = 1.0,
) -> torch.Tensor:
    """在归一化评分单位计算带样本权重的 Smooth L1 损失。

    参数:
        predictions: 网络预测，形状为 ``[batch]``。
        batch: 含目标值和正数样本权重的训练批次。
        beta: Smooth L1 从二次项切换到一次项的误差阈值。
        score_scale: 一个原始网络输出单位对应的引擎分值。

    返回:
        按样本权重归一化后的标量损失。
    """

    losses = F.smooth_l1_loss(
        predictions, batch.targets / score_scale,
        reduction="none", beta=beta / score_scale,
    )
    return torch.sum(losses * batch.weights) / torch.sum(batch.weights)


def combined_loss(
    predictions: torch.Tensor,
    batch: NnueBatch,
    beta: float,
    outcome_weight: float,
    outcome_scale: float,
    score_scale: float = 1.0,
) -> torch.Tensor:
    """拟合所有评分，同时仅对有赛果样本拟合胜率信号。

    网络只输出一个当前方视角的分数；用 sigmoid(score / outcome_scale)
    映射到胜率代理，胜/和/负的目标分别为 1、0.5、0。赛果项按整批样本
    权重归一化，避免只有一条有赛果的批次被意外放大。
    """

    score_loss = weighted_huber_loss(predictions, batch, beta, score_scale)
    if outcome_weight == 0.0 or not bool(torch.any(batch.result_known)):
        return score_loss
    known = batch.result_known
    outcome_targets = (batch.results[known] + 1.0) / 2.0
    outcome_losses = F.binary_cross_entropy_with_logits(
        predictions[known] * (score_scale / outcome_scale),
        outcome_targets, reduction="none",
    )
    outcome_loss = torch.sum(outcome_losses * batch.weights[known]) / torch.sum(
        batch.weights
    )
    return score_loss + outcome_weight * outcome_loss


def run_epoch(
    model: HalfKANetwork,
    loader: DataLoader[NnueBatch],
    device: torch.device,
    beta: float,
    outcome_weight: float,
    outcome_scale: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    collect_metrics: bool = False,
) -> float | dict[str, Any]:
    """运行一轮训练或验证。

    参数:
        model: 要训练或验证的 HalfKA 网络。
        loader: 产生 ``NnueBatch`` 的数据加载器。
        device: 计算使用的设备。
        beta: Smooth L1 损失阈值。
        outcome_weight: 赛果损失相对于评分损失的权重。
        outcome_scale: 分数映射为胜率代理时的尺度。
        optimizer: 训练时使用的优化器；传入 ``None`` 表示只验证。
        gradient_clip: 梯度范数上限，防止异常标签造成梯度爆炸。

    返回:
        默认返回批次损失平均值；``collect_metrics`` 为真时返回详细轮次指标。

    异常:
        RuntimeError: 数据加载器没有产生任何批次。
    """

    training = optimizer is not None
    model.train(training)
    metrics = (
        EpochMetrics(
            score_scale=model.score_scale,
            huber_beta=beta,
            outcome_scale=outcome_scale,
            outcome_weight=outcome_weight,
        )
        if collect_metrics
        else None
    )
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            if metrics is not None:
                predictions, accumulator_pre, hidden_pre = model.forward_with_activations(
                    batch.red_indices,
                    batch.red_offsets,
                    batch.black_indices,
                    batch.black_offsets,
                    batch.red_to_move,
                )
            else:
                predictions = model.forward_batch(batch)
            loss = combined_loss(
                predictions, batch, beta, outcome_weight, outcome_scale,
                model.score_scale,
            )
        if metrics is not None:
            metrics.update(predictions, batch, accumulator_pre, hidden_pre)
        if training:
            # set_to_none 避免把大尺寸第一层梯度缓冲区逐项清零。
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("data loader produced no batches")
    if metrics is not None:
        return metrics.finish()
    return total_loss / batches


def _sample_digest(samples: list[TrainingSample]) -> str:
    """散列一组样本身份和监督值，供续训兼容性检查。"""
    rows = [
        (
            sample.group_key,
            sample.position_key,
            sample.target,
            sample.weight,
            sample.result,
            sample.phase,
            sample.source_fingerprint,
        )
        for sample in samples
    ]
    rows.sort(
        key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    )
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_metadata(
    dataset: Dataset[TrainingSample],
    train_loader: DataLoader[NnueBatch],
    validation_loader: DataLoader[NnueBatch] | None,
    test_loader: DataLoader[NnueBatch] | None,
    dropped: int,
    *,
    validation_fraction: float,
    seed: int,
    test_manifest: Path,
    fixed_test_samples: list[TrainingSample] | None = None,
) -> dict[str, Any]:
    """记录真实训练/验证/固定测试划分，避免静默更换续训数据。"""

    samples = [dataset[index] for index in range(len(dataset))]

    def indices(loader: DataLoader[NnueBatch] | None) -> list[int]:
        if loader is None:
            return []
        subset = loader.dataset
        if not isinstance(subset, Subset):
            raise TypeError("split loader dataset must be a torch Subset")
        return [int(index) for index in subset.indices]

    partition_indices = {
        "train": indices(train_loader),
        "validation": indices(validation_loader),
        "test": [],
    }
    partition_samples = {
        name: [samples[index] for index in values]
        for name, values in partition_indices.items()
    }
    if fixed_test_samples is not None:
        partition_samples["test"] = fixed_test_samples
    manifest_bytes = test_manifest.read_bytes()
    return {
        "version": 1,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "test_manifest": {
            "path": str(test_manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "dropped_duplicate_count": dropped,
        "counts": {name: len(values) for name, values in partition_samples.items()},
        "known_result_counts": {
            name: sum(sample.result is not None for sample in values)
            for name, values in partition_samples.items()
        },
        "sample_digests": {
            name: _sample_digest(values) for name, values in partition_samples.items()
        },
    }


def load_fixed_test_data(
    manifest_path: Path,
    quarantine_sources: set[str],
    narrow_score_limit: float,
    score_clip: float,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader[NnueBatch] | None, list[TrainingSample]]:
    """从固定 manifest 的原始源独立加载测试样本，避免 generation replay 缺源。"""

    manifest = load_test_manifest(manifest_path)
    test_groups = set(manifest["groups"])
    source_ids = {group.split("#", 1)[0] for group in test_groups}
    source_fingerprints = manifest.get("source_fingerprints")
    if source_fingerprints is not None:
        if not isinstance(source_fingerprints, dict):
            raise ValueError("fixed-test source_fingerprints must be an object")
        missing = source_ids - set(source_fingerprints)
        if missing:
            raise ValueError(
                "fixed-test manifest lacks source fingerprints for: "
                + ", ".join(sorted(missing))
            )

    sources: list[JsonlPositionDataset] = []
    for identifier in sorted(source_ids):
        source_path = manifest_source_path(identifier)
        if not source_path.is_file():
            raise FileNotFoundError(f"fixed-test source is missing: {source_path}")
        if source_fingerprints is not None:
            actual = file_fingerprint(source_path)
            if actual != source_fingerprints[identifier]:
                raise ValueError(
                    f"fixed-test source changed since manifest creation: {source_path}"
                )
        sources.append(
            JsonlPositionDataset(
                source_path,
                score_clip=score_clip,
                quarantine_sources=quarantine_sources,
                narrow_score_limit=narrow_score_limit,
                allow_empty=True,
                group_filter=test_groups,
            )
        )

    anchor_dataset: Dataset[TrainingSample] | None = None
    if sources:
        nonempty = [source for source in sources if len(source)]
        if nonempty:
            anchor_dataset = (
                nonempty[0]
                if len(nonempty) == 1
                else ConcatDataset(nonempty)
            )
    if anchor_dataset is None:
        raise ValueError("fixed-test manifest sources contain no usable samples")

    all_samples = [anchor_dataset[index] for index in range(len(anchor_dataset))]
    selected_indices: list[int] = []
    seen_test_positions: set[str] = set()
    for index, sample in enumerate(all_samples):
        if sample.group_key not in test_groups or sample.position_key in seen_test_positions:
            continue
        selected_indices.append(index)
        seen_test_positions.add(sample.position_key)
    selected_samples = [all_samples[index] for index in selected_indices]
    if not selected_samples:
        raise ValueError("fixed-test manifest groups do not match their source data")
    missing_positions = set(manifest["position_keys"]) - {
        sample.position_key for sample in selected_samples
    }
    if missing_positions:
        raise ValueError(
            "fixed-test source is missing manifest positions after filtering/quarantine"
        )
    loader: DataLoader[NnueBatch] = DataLoader(
        Subset(anchor_dataset, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_samples,
        persistent_workers=workers > 0,
    )
    return loader, selected_samples


def _training_args(args: argparse.Namespace) -> dict[str, Any]:
    """把 argparse 参数转换为可 JSON 序列化的配置。"""

    return {
        key: (
            [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }


def validate_resume_split(
    checkpoint: dict[str, Any],
    current_split: dict[str, Any],
    *,
    reset_best: bool,
) -> tuple[bool, str | None]:
    """拒绝未经确认的数据划分变化，并说明是否应重置优化器状态。"""

    previous_split = checkpoint.get("split_metadata")
    compatible = previous_split == current_split
    if previous_split is not None and not compatible and not reset_best:
        raise ValueError(
            "resume checkpoint uses a different data split; pass --reset-best "
            "to deliberately start a new validation baseline"
        )
    if previous_split is None:
        return False, "checkpoint has no split compatibility metadata"
    if not compatible:
        return False, "data split changed"
    return True, None


def make_loaders(
    dataset: Dataset[TrainingSample],
    batch_size: int,
    validation_fraction: float,
    workers: int,
    seed: int,
    test_manifest: Path | None = None,
    split_seed: int | None = None,
) -> tuple[
    DataLoader[NnueBatch], DataLoader[NnueBatch] | None,
    DataLoader[NnueBatch] | None, int,
]:
    """按文件加对局 ID 分组，并隔离固定测试集和镜像重复局面。

    参数:
        dataset: 已编码训练数据集。
        batch_size: 每批样本数。
        validation_fraction: 留作验证集的比例，范围为 ``[0, 1)``。
        workers: DataLoader 子进程数量。
        seed: 数据拆分和训练集洗牌使用的随机种子。

    返回:
        ``(training_loader, validation_loader, test_loader, dropped)``。
    """

    samples = [dataset[index] for index in range(len(dataset))]
    manifest = load_test_manifest(test_manifest) if test_manifest else None
    train_indices, validation_indices, test_indices, dropped = split_indices(
        samples, validation_fraction, seed if split_seed is None else split_seed, manifest
    )
    if not train_indices:
        raise ValueError("no training samples remain after fixed-test isolation")
    training_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices) if validation_indices else None
    test_set = Subset(dataset, test_indices) if test_indices else None
    generator = torch.Generator().manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "collate_fn": collate_samples,
        "persistent_workers": workers > 0,
    }
    training_loader: DataLoader[NnueBatch] = DataLoader(
        training_set, shuffle=True, generator=generator, **common
    )
    validation_loader = (
        DataLoader(validation_set, shuffle=False, **common)
        if validation_set is not None
        else None
    )
    test_loader = (
        DataLoader(test_set, shuffle=False, **common) if test_set is not None else None
    )
    return training_loader, validation_loader, test_loader, dropped


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数并返回 argparse 命名空间。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="+",
        help="one or more JSONL files with fen and score fields",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs/default"))
    parser.add_argument("--resume", type=Path, help="resume from a .pt checkpoint")
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="reset the best validation loss after loading a checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST,
        help="permanent held-out game and mirrored-position manifest",
    )
    parser.add_argument(
        "--quarantine-manifest", type=Path, default=DEFAULT_QUARANTINE_MANIFEST,
        help="known legacy files whose narrow score labels must not enter training",
    )
    parser.add_argument("--huber-beta", type=float, default=100.0)
    parser.add_argument(
        "--score-scale", type=float, default=DEFAULT_SCORE_SCALE,
        help="engine score units represented by one raw network output unit",
    )
    parser.add_argument(
        "--outcome-weight", type=float, default=0.0,
        help="weight for known-result BCE loss; default 0 preserves score-only training",
    )
    parser.add_argument(
        "--outcome-scale", type=float, default=600.0,
        help="score scale for the sigmoid used by the known-result loss",
    )
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--score-clip", type=float, default=28_000.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026, help="model/shuffle seed")
    parser.add_argument(
        "--split-seed", type=int, default=2026,
        help="stable seed for the train/validation split; independent of training randomness",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args()


def main() -> None:
    """执行数据加载、可选续训、逐轮训练以及最佳/最新检查点保存。"""

    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if args.outcome_weight < 0.0 or args.outcome_scale <= 0.0 or args.score_scale <= 0.0:
        raise ValueError("outcome weight must be nonnegative and scale positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if not args.quarantine_manifest.is_file():
        raise FileNotFoundError(f"score quarantine manifest missing: {args.quarantine_manifest}")
    quarantine_sources, narrow_score_limit = load_quarantine_manifest(
        args.quarantine_manifest
    )
    loaded_datasets = [
        JsonlPositionDataset(
            path, score_clip=args.score_clip,
            quarantine_sources=quarantine_sources,
            narrow_score_limit=narrow_score_limit, allow_empty=True,
        )
        for path in args.dataset
    ]
    quarantined_count = sum(source.quarantined_count for source in loaded_datasets)
    datasets = [source for source in loaded_datasets if len(source) > 0]
    for source in datasets:
        if len(source) >= 100 and source.maximum_absolute_score <= narrow_score_limit:
            raise ValueError(
                f"suspicious narrow score labels in {source.path}: all {len(source)} "
                f"samples are within +/-{narrow_score_limit:g}; relabel before training"
            )
    if not datasets:
        raise ValueError("all training samples were quarantined")
    dataset: Dataset[TrainingSample] = (
        datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    )
    known_results = sum(
        sample.result is not None for source in datasets for sample in source.samples
    )
    if not args.test_manifest.is_file():
        raise FileNotFoundError(
            f"fixed test manifest missing: {args.test_manifest}; "
            "create it with python -m training.partition"
        )
    train_loader, validation_loader, _, dropped = make_loaders(
        dataset,
        args.batch_size,
        args.validation_fraction,
        args.workers,
        args.seed,
        args.test_manifest,
        args.split_seed,
    )
    test_loader, fixed_test_samples = load_fixed_test_data(
        args.test_manifest,
        quarantine_sources,
        narrow_score_limit,
        args.score_clip,
        args.batch_size,
        args.workers,
    )

    current_split = split_metadata(
        dataset,
        train_loader,
        validation_loader,
        test_loader,
        dropped,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
        test_manifest=args.test_manifest,
        fixed_test_samples=fixed_test_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "format": "xiangqi-nnue-training-config-v1",
        "training_args": _training_args(args),
        "score_units": {
            "score_scale_engine_units_per_raw_output": args.score_scale,
            "huber_beta_engine_units": args.huber_beta,
            "outcome_scale_engine_units": args.outcome_scale,
        },
        "dataset": {
            "paths": [str(path.resolve()) for path in args.dataset],
            "sample_count": len(dataset),
            "result_known_count": known_results,
            "result_unknown_count": len(dataset) - known_results,
            "quarantined_count": quarantined_count,
        },
        "split": current_split,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    model = HalfKANetwork(args.score_scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    resumed_checkpoint: dict[str, Any] | None = None
    reset_optimizer_reason: str | None = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        resumed_checkpoint = checkpoint
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        previous_scale = checkpoint_score_scale(checkpoint)
        split_is_compatible, reset_optimizer_reason = validate_resume_split(
            checkpoint, current_split, reset_best=args.reset_best
        )
        if previous_scale != args.score_scale:
            # 输出层由旧单位换算到新单位。旧 AdamW 动量/方差的单位不同，
            # 不直接沿用，以免续训时第一步发生不合理更新。
            migrate_output_score_scale(model, previous_scale, args.score_scale)
            print(
                f"migrated output scale {previous_scale:g} -> {args.score_scale:g}; "
                "reset optimizer state",
                flush=True,
            )
            reset_optimizer_reason = "score scale changed"
        elif split_is_compatible and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif reset_optimizer_reason:
            print(f"reset optimizer state: {reset_optimizer_reason}", flush=True)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if split_is_compatible and not args.reset_best:
            best_validation_loss = float(
                checkpoint.get("best_validation_loss", float("inf"))
            )
            best_epoch = checkpoint.get("best_epoch")

    write_json(args.output_dir / "config.json", run_config)
    print(
        f"device={device} samples={len(dataset)} "
        f"train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset) if validation_loader else 0} "
        f"test={len(test_loader.dataset) if test_loader else 0} "
        f"deduplicated={dropped} quarantined={quarantined_count} "
        f"known_results={known_results} "
        f"unknown_results={len(dataset) - known_results}"
    )
    metrics_path = args.output_dir / "metrics.jsonl"

    if resumed_checkpoint is not None:
        # 恢复时用当前固定验证划分重新测量历史 best。这样换尺度时也能无损迁移
        # 末层参数，同时不会把旧划分的 best_loss 带到新划分中。
        baseline_source = resumed_checkpoint
        source_best_path = args.resume.parent / "best.pt"
        if source_best_path.is_file() and not args.reset_best:
            possible_best = load_checkpoint(source_best_path, device)
            if possible_best.get("split_metadata") == current_split:
                baseline_source = possible_best
        baseline_scale = checkpoint_score_scale(baseline_source)
        baseline_model = HalfKANetwork(baseline_scale).to(device)
        baseline_model.load_state_dict(
            baseline_source["model_state_dict"], strict=True
        )
        if baseline_scale != args.score_scale:
            migrate_output_score_scale(
                baseline_model, baseline_scale, args.score_scale
            )
        baseline_loader = validation_loader or train_loader
        baseline_metrics = run_epoch(
            baseline_model,
            baseline_loader,
            device,
            args.huber_beta,
            args.outcome_weight,
            args.outcome_scale,
            None,
            args.gradient_clip,
            collect_metrics=True,
        )
        assert isinstance(baseline_metrics, dict)
        best_validation_loss = float(baseline_metrics["total_loss"])
        best_epoch = int(baseline_source.get("epoch", start_epoch - 1))
        save_checkpoint(
            args.output_dir / "best.pt",
            {
                "model_state_dict": baseline_model.state_dict(),
                "epoch": best_epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "validation_metrics": baseline_metrics,
                "split_metadata": current_split,
                "training_args": _training_args(args),
            },
            score_scale=args.score_scale,
        )
        append_jsonl(
            metrics_path,
            {
                "event": "resume_baseline",
                "epoch": best_epoch,
                "source": str(args.resume.resolve()),
                "validation": baseline_metrics,
                "optimizer_reset_reason": reset_optimizer_reason,
            },
        )

    for epoch in range(start_epoch, args.epochs + 1):
        training_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.huber_beta,
            args.outcome_weight,
            args.outcome_scale,
            optimizer,
            args.gradient_clip,
            collect_metrics=True,
        )
        assert isinstance(training_metrics, dict)
        validation_metrics = (
            run_epoch(
                model,
                validation_loader,
                device,
                args.huber_beta,
                args.outcome_weight,
                args.outcome_scale,
                None,
                args.gradient_clip,
                collect_metrics=True,
            )
            if validation_loader
            else training_metrics
        )
        assert isinstance(validation_metrics, dict)
        training_loss = float(training_metrics["total_loss"])
        validation_loss = float(validation_metrics["total_loss"])
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "split_metadata": current_split,
            "training_args": _training_args(args),
        }
        # latest 每轮覆盖，best 只在验证损失改善时更新。
        save_checkpoint(args.output_dir / "latest.pt", payload, score_scale=args.score_scale)
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                    "validation_metrics": validation_metrics,
                    "split_metadata": current_split,
                    "training_args": _training_args(args),
                },
                score_scale=args.score_scale,
            )
        append_jsonl(
            metrics_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "training": training_metrics,
                "validation": validation_metrics,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            },
        )
        print(
            f"epoch={epoch} train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

    fixed_test_metrics: dict[str, Any] | None = None
    if test_loader is not None:
        best_checkpoint = load_checkpoint(args.output_dir / "best.pt", device)
        best_model = HalfKANetwork(checkpoint_score_scale(best_checkpoint)).to(device)
        best_model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        fixed_test_metrics = run_epoch(
            best_model, test_loader, device, args.huber_beta,
            args.outcome_weight, args.outcome_scale, None, args.gradient_clip,
            collect_metrics=True,
        )
        assert isinstance(fixed_test_metrics, dict)
        append_jsonl(
            metrics_path,
            {
                "event": "fixed_test",
                "selected_best_epoch": best_checkpoint.get("best_epoch"),
                "metrics": fixed_test_metrics,
            },
        )
        print(f"fixed_test_loss={fixed_test_metrics['total_loss']:.6f}")

    write_json(
        args.output_dir / "report.json",
        {
            "format": "xiangqi-nnue-training-report-v1",
            "config": run_config,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "fixed_test_evaluations": 1 if fixed_test_metrics is not None else 0,
            "fixed_test": fixed_test_metrics,
        },
    )


if __name__ == "__main__":
    main()
