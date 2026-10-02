"""保存、加载并校验带版本信息的 PyTorch NNUE 训练检查点。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .halfka import FEATURE_DIMENSIONS
from .model import (
    ACCUMULATOR_SIZE,
    ARCHITECTURE_NAME,
    HIDDEN_SIZE,
    HalfKANetwork,
)

CHECKPOINT_FORMAT = "xiangqi-halfka-training-v1"


def architecture_metadata() -> dict[str, int | str]:
    """返回用于拒绝不兼容检查点的固定网络结构元数据。"""

    return {
        "name": ARCHITECTURE_NAME,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "accumulator_size": ACCUMULATOR_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """检查训练格式、网络维度和模型参数字段。

    参数:
        checkpoint: ``torch.load`` 得到的检查点字典。

    异常:
        ValueError: 格式版本、网络结构不匹配或缺少模型参数。
    """

    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported training checkpoint format")
    if checkpoint.get("architecture") != architecture_metadata():
        raise ValueError("checkpoint architecture does not match this engine")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """加载并校验本项目生成的可信检查点。

    参数:
        path: ``.pt`` 检查点路径。
        device: 加载张量的目标设备。

    返回:
        已通过格式和结构校验的检查点字典。

    异常:
        OSError: 文件无法读取。
        ValueError: 根对象或检查点格式不合法。
    """

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint root must be a dictionary")
    validate_checkpoint(checkpoint)
    return checkpoint


def load_model(path: str | Path, device: str | torch.device = "cpu") -> HalfKANetwork:
    """从检查点构造并恢复一个 ``HalfKANetwork``。

    参数:
        path: ``.pt`` 检查点路径。
        device: 模型和参数的目标设备。

    返回:
        严格加载全部参数后的网络。
    """

    checkpoint = load_checkpoint(path, device)
    model = HalfKANetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    """自动补充格式元数据并原子保存训练检查点。

    参数:
        path: 目标 ``.pt`` 路径；父目录不存在时会自动创建。
        checkpoint: 至少包含 ``model_state_dict`` 的训练状态。

    说明:
        先写入同目录临时文件，再用 ``os.replace`` 替换目标，避免中断时留下半个文件。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload["format"] = CHECKPOINT_FORMAT
    payload["architecture"] = architecture_metadata()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
