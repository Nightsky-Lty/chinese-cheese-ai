"""保存、加载并校验带版本信息的 PyTorch NNUE 训练检查点。"""

from __future__ import annotations

import os
import math
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

CHECKPOINT_FORMAT = "xiangqi-halfka-training-v2"
LEGACY_CHECKPOINT_FORMAT = "xiangqi-halfka-training-v1"
SCORE_PERSPECTIVE = "side_to_move"


def checkpoint_score_scale(checkpoint: dict[str, Any]) -> float:
    """读取末层原始输出到引擎分数的倍数；旧格式始终为 1。"""

    if checkpoint.get("format") == LEGACY_CHECKPOINT_FORMAT:
        return 1.0
    scoring = checkpoint.get("scoring")
    if not isinstance(scoring, dict) or scoring.get("version") != 1:
        raise ValueError("checkpoint lacks supported scoring metadata")
    if scoring.get("perspective") != SCORE_PERSPECTIVE:
        raise ValueError("checkpoint score perspective is incompatible")
    scale = scoring.get("output_to_engine_score")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError("checkpoint score scale is invalid")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("checkpoint score scale must be finite and positive")
    return scale


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

    if checkpoint.get("format") not in (CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT):
        raise ValueError("unsupported training checkpoint format")
    if checkpoint.get("architecture") != architecture_metadata():
        raise ValueError("checkpoint architecture does not match this engine")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    checkpoint_score_scale(checkpoint)


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
    model = HalfKANetwork(checkpoint_score_scale(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def migrate_output_score_scale(
    model: HalfKANetwork, previous_scale: float, target_scale: float
) -> None:
    """Rescale only the output layer while preserving engine-unit predictions.

    The checkpoint's stored raw output is in ``1 / previous_scale`` engine units.
    Multiplying its final linear layer by ``previous_scale / target_scale`` and
    updating the model metadata therefore changes the raw unit without changing
    the represented engine score. Optimizer state is intentionally managed by the
    caller because its moments use the old parameter scale.
    """

    for value in (previous_scale, target_scale):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("score scales must be numeric")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("score scales must be finite and positive")
    ratio = float(previous_scale) / float(target_scale)
    if ratio != 1.0:
        with torch.no_grad():
            model.output.weight.mul_(ratio)
            model.output.bias.mul_(ratio)
    model.score_scale = float(target_scale)


def save_checkpoint(
    path: str | Path, checkpoint: dict[str, Any], *, score_scale: float
) -> None:
    """自动补充格式元数据并原子保存训练检查点。

    参数:
        path: 目标 ``.pt`` 路径；父目录不存在时会自动创建。
        checkpoint: 至少包含 ``model_state_dict`` 的训练状态。
        score_scale: 末层原始输出转换为引擎分数时的倍数。

    说明:
        先写入同目录临时文件，再用 ``os.replace`` 替换目标，避免中断时留下半个文件。
    """

    destination = Path(path)
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("score scale must be finite and positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload["format"] = CHECKPOINT_FORMAT
    payload["architecture"] = architecture_metadata()
    payload["scoring"] = {
        "version": 1,
        "perspective": SCORE_PERSPECTIVE,
        "output_to_engine_score": float(score_scale),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
