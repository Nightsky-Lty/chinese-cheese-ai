"""把 PyTorch 检查点导出为 C++ 可直接加载的 ``XQNNUEF1`` 格式。"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .halfka import FEATURE_DIMENSIONS
from .model import ACCUMULATOR_SIZE, HIDDEN_SIZE, HalfKANetwork

FILE_MAGIC = b"XQNNUEF1"
FILE_VERSION = 1
HEADER = struct.Struct("<8sIIII")


def _float32_bytes(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> bytes:
    """校验张量并转换为连续的小端 float32 字节。

    参数:
        tensor: 要导出的模型参数。
        expected_shape: C++ 文件格式要求的精确形状。

    返回:
        按 C 行优先顺序排列的小端 float32 字节。

    异常:
        ValueError: 参数形状不匹配或包含 NaN/无穷大。
    """

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"parameter shape {tuple(tensor.shape)} does not match {expected_shape}"
        )
    value = tensor.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError("cannot export NaN or infinity in NNUE parameters")
    array = np.asarray(value.numpy(), dtype="<f4", order="C")
    return array.tobytes(order="C")


def export_model(model: HalfKANetwork, destination: str | Path) -> Path:
    """按照 C++ 读取顺序原子写出一个推理模型。

    参数:
        model: 已训练的 ``HalfKANetwork``。
        destination: 目标 ``.nnue`` 文件路径。

    返回:
        已写入文件的 ``Path``。

    说明:
        参数顺序必须保持为特征偏置、特征权重、隐藏偏置、隐藏权重、输出偏置、
        输出权重。先写临时文件并 ``fsync``，完成后再替换目标文件。
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    model = model.to("cpu").eval()
    with temporary.open("wb") as stream:
        stream.write(
            HEADER.pack(
                FILE_MAGIC,
                FILE_VERSION,
                FEATURE_DIMENSIONS,
                ACCUMULATOR_SIZE,
                HIDDEN_SIZE,
            )
        )
        # 以下顺序与 NnueNetwork::load() 完全一致，不能随意调整。
        stream.write(_float32_bytes(model.feature_bias, (ACCUMULATOR_SIZE,)))
        stream.write(
            _float32_bytes(
                model.feature_weights,
                (FEATURE_DIMENSIONS, ACCUMULATOR_SIZE),
            )
        )
        stream.write(_float32_bytes(model.hidden.bias, (HIDDEN_SIZE,)))
        stream.write(
            _float32_bytes(
                model.hidden.weight,
                (HIDDEN_SIZE, ACCUMULATOR_SIZE * 2),
            )
        )
        stream.write(_float32_bytes(model.output.bias, (1,)))
        stream.write(_float32_bytes(model.output.weight, (1, HIDDEN_SIZE)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def export_checkpoint(checkpoint: str | Path, destination: str | Path) -> Path:
    """从训练检查点加载模型并只导出推理所需参数。

    参数:
        checkpoint: 输入 ``.pt`` 检查点。
        destination: 输出 ``.nnue`` 路径。

    返回:
        已导出模型的路径。
    """

    return export_model(load_model(checkpoint, "cpu"), destination)


def parse_args() -> argparse.Namespace:
    """解析检查点路径和导出目标路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="input training checkpoint (.pt)")
    parser.add_argument("destination", type=Path, help="output C++ model (.nnue)")
    return parser.parse_args()


def main() -> None:
    """执行命令行导出并报告最终文件大小。"""

    args = parse_args()
    destination = export_checkpoint(args.checkpoint, args.destination)
    print(f"exported {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
