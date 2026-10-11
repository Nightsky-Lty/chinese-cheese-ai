"""保存、加载并校验带版本信息的 PyTorch NNUE 训练检查点。"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import torch

from .halfka import FEATURE_DIMENSIONS
from .model import (
    HIDDEN_SIZE,
    HalfKANetwork,
    NetworkConfig,
)

CHECKPOINT_FORMAT = "xiangqi-halfka-training-v2"
CONFIGURABLE_CHECKPOINT_FORMAT = "xiangqi-halfka-training-v3"
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


def architecture_metadata(config: NetworkConfig = NetworkConfig()) -> dict[str, Any]:
    """记录结构；默认架构保留旧版元数据，扩展架构记录完整隐藏层列表。"""

    return {
        "name": config.name,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "accumulator_size": config.accumulator_size,
        **({"hidden_size": HIDDEN_SIZE} if config == NetworkConfig()
           else {"hidden_sizes": list(config.hidden_sizes)}),
    }


def checkpoint_network_config(checkpoint: dict[str, Any]) -> NetworkConfig:
    """由已存储的版本和结构恢复配置，旧格式只能表示原有默认架构。"""
    metadata = checkpoint.get("architecture")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint lacks architecture metadata")
    if checkpoint.get("format") != CONFIGURABLE_CHECKPOINT_FORMAT:
        if metadata != architecture_metadata():
            raise ValueError("legacy checkpoint architecture is incompatible")
        return NetworkConfig()
    hidden = metadata.get("hidden_sizes")
    if not isinstance(hidden, list):
        raise ValueError("checkpoint lacks hidden layer sizes")
    config = NetworkConfig(metadata.get("accumulator_size"), tuple(hidden))
    if metadata != architecture_metadata(config):
        raise ValueError("checkpoint architecture metadata is inconsistent")
    return config


def model_from_checkpoint(checkpoint: dict[str, Any], device: str | torch.device = "cpu") -> HalfKANetwork:
    """按照检查点尺寸严格恢复模型；不自动把旧权重填充到新架构。"""
    validate_checkpoint(checkpoint)
    config = checkpoint_network_config(checkpoint)
    model = HalfKANetwork(checkpoint_score_scale(checkpoint),
                          accumulator_size=config.accumulator_size,
                          hidden_sizes=config.hidden_sizes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """检查训练格式、网络维度和模型参数字段。

    参数:
        checkpoint: ``torch.load`` 得到的检查点字典。

    异常:
        ValueError: 格式版本、网络结构不匹配或缺少模型参数。
    """

    if checkpoint.get("format") not in (CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT,
                                       CONFIGURABLE_CHECKPOINT_FORMAT):
        raise ValueError("unsupported training checkpoint format")
    config = checkpoint_network_config(checkpoint)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    expected = {
        "feature_weights": (FEATURE_DIMENSIONS, config.accumulator_size),
        "feature_bias": (config.accumulator_size,),
        "output.weight": (1, config.hidden_sizes[-1]),
        "output.bias": (1,),
    }
    input_size = config.accumulator_size * 2
    for index, width in enumerate(config.hidden_sizes):
        prefix = "hidden" if index == 0 else f"extra_hidden.{index - 1}"
        expected[f"{prefix}.weight"] = (width, input_size)
        expected[f"{prefix}.bias"] = (width,)
        input_size = width
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names do not match architecture")
    for name, shape in expected.items():
        if not isinstance(state[name], torch.Tensor) or tuple(state[name].shape) != shape:
            raise ValueError(f"checkpoint parameter shape does not match architecture: {name}")
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
    return model_from_checkpoint(checkpoint, device)


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
    path: str | Path, checkpoint: dict[str, Any], *, score_scale: float,
    network_config: NetworkConfig = NetworkConfig(),
) -> None:
    """自动补充格式元数据并原子保存训练检查点。

    参数:
        path: 目标 ``.pt`` 路径；父目录不存在时会自动创建。
        checkpoint: 至少包含 ``model_state_dict`` 的训练状态。
        score_scale: 末层原始输出转换为引擎分数时的倍数。
        network_config: 模型的完整层尺寸；未传入时保持原有默认架构。

    说明:
        先写入同目录临时文件，再用 ``os.replace`` 替换目标，避免中断时留下半个文件。
    """

    destination = Path(path)
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("score scale must be finite and positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload["format"] = (CHECKPOINT_FORMAT if network_config == NetworkConfig()
                         else CONFIGURABLE_CHECKPOINT_FORMAT)
    payload["architecture"] = architecture_metadata(network_config)
    payload["scoring"] = {
        "version": 1,
        "perspective": SCORE_PERSPECTIVE,
        "output_to_engine_score": float(score_scale),
    }
    validate_checkpoint(payload)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
