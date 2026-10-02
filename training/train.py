"""使用当前行棋方视角的局面分标签训练 HalfKA NNUE。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import JsonlPositionDataset, TrainingSample, collate_samples
from .model import HalfKANetwork, NnueBatch


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


def weighted_huber_loss(predictions: torch.Tensor, batch: NnueBatch, beta: float) -> torch.Tensor:
    """计算以引擎分值为单位、带样本权重的 Smooth L1 损失。

    参数:
        predictions: 网络预测，形状为 ``[batch]``。
        batch: 含目标值和正数样本权重的训练批次。
        beta: Smooth L1 从二次项切换到一次项的误差阈值。

    返回:
        按样本权重归一化后的标量损失。
    """

    losses = F.smooth_l1_loss(
        predictions, batch.targets, reduction="none", beta=beta
    )
    return torch.sum(losses * batch.weights) / torch.sum(batch.weights)


def run_epoch(
    model: HalfKANetwork,
    loader: DataLoader[NnueBatch],
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> float:
    """运行一轮训练或验证。

    参数:
        model: 要训练或验证的 HalfKA 网络。
        loader: 产生 ``NnueBatch`` 的数据加载器。
        device: 计算使用的设备。
        beta: Smooth L1 损失阈值。
        optimizer: 训练时使用的优化器；传入 ``None`` 表示只验证。
        gradient_clip: 梯度范数上限，防止异常标签造成梯度爆炸。

    返回:
        所有批次损失的算术平均值。

    异常:
        RuntimeError: 数据加载器没有产生任何批次。
    """

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            predictions = model.forward_batch(batch)
            loss = weighted_huber_loss(predictions, batch, beta)
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
    return total_loss / batches


def make_loaders(
    dataset: Dataset[TrainingSample],
    batch_size: int,
    validation_fraction: float,
    workers: int,
    seed: int,
) -> tuple[DataLoader[NnueBatch], DataLoader[NnueBatch] | None]:
    """按固定随机种子拆分数据并创建训练/验证加载器。

    参数:
        dataset: 已编码训练数据集。
        batch_size: 每批样本数。
        validation_fraction: 留作验证集的比例，范围为 ``[0, 1)``。
        workers: DataLoader 子进程数量。
        seed: 数据拆分和训练集洗牌使用的随机种子。

    返回:
        ``(training_loader, validation_loader)``；验证比例为零时第二项为空。
    """

    validation_size = int(round(len(dataset) * validation_fraction))
    if validation_fraction > 0.0 and len(dataset) > 1:
        validation_size = max(1, min(validation_size, len(dataset) - 1))
    else:
        validation_size = 0
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    if validation_size:
        training_set, validation_set = random_split(
            dataset, [training_size, validation_size], generator=generator
        )
    else:
        training_set = dataset
        validation_set = None

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
    return training_loader, validation_loader


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数并返回 argparse 命名空间。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL file with fen and score fields")
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs/default"))
    parser.add_argument("--resume", type=Path, help="resume from a .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=100.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--score-clip", type=float, default=28_000.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args()


def main() -> None:
    """执行数据加载、可选续训、逐轮训练以及最佳/最新检查点保存。"""

    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dataset = JsonlPositionDataset(args.dataset, score_clip=args.score_clip)
    train_loader, validation_loader = make_loaders(
        dataset,
        args.batch_size,
        args.validation_fraction,
        args.workers,
        args.seed,
    )

    model = HalfKANetwork().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 1
    best_validation_loss = float("inf")
    if args.resume:
        # 模型与优化器一起恢复，确保 AdamW 的动量不会在续训时丢失。
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} samples={len(dataset)} "
        f"train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset) if validation_loader else 0}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        training_loss = run_epoch(
            model,
            train_loader,
            device,
            args.huber_beta,
            optimizer,
            args.gradient_clip,
        )
        validation_loss = (
            run_epoch(
                model,
                validation_loader,
                device,
                args.huber_beta,
                None,
                args.gradient_clip,
            )
            if validation_loader
            else training_loss
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": min(best_validation_loss, validation_loss),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        # latest 每轮覆盖，best 只在验证损失改善时更新。
        save_checkpoint(args.output_dir / "latest.pt", payload)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            payload["best_validation_loss"] = best_validation_loss
            save_checkpoint(args.output_dir / "best.pt", payload)
        print(
            f"epoch={epoch} train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )


if __name__ == "__main__":
    main()
