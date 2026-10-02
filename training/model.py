"""定义与 C++ ``NnueNetwork`` 参数布局一致的 PyTorch 网络。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .halfka import FEATURE_DIMENSIONS

ACCUMULATOR_SIZE = 256
CONCATENATED_SIZE = ACCUMULATOR_SIZE * 2
HIDDEN_SIZE = 32
EVALUATION_LIMIT = 28_000
ARCHITECTURE_NAME = "halfka-11340-256x2-32-1-f32"


@dataclass
class NnueBatch:
    """一批局面的紧凑稀疏特征和监督标签。

    ``*_indices`` 连续保存整批激活特征，``*_offsets`` 保存每个样本在连续数组中的
    起止边界。例如 offsets 为 ``[0, 32, 60]`` 时，前 32 项属于样本 0，之后
    28 项属于样本 1。这样无需创建巨大的稠密 0/1 输入。

    属性:
        red_indices: 红方视角的扁平特征编号，类型必须为 ``torch.long``。
        red_offsets: 红方特征分段边界，长度为批大小加一。
        black_indices: 黑方视角的扁平特征编号。
        black_offsets: 黑方特征分段边界。
        red_to_move: 每个样本是否由红方行棋的布尔张量。
        targets: 当前行棋方视角的目标引擎分值。
        weights: 每条样本的正数损失权重。
    """

    red_indices: Tensor
    red_offsets: Tensor
    black_indices: Tensor
    black_offsets: Tensor
    red_to_move: Tensor
    targets: Tensor
    weights: Tensor

    def to(self, device: torch.device | str) -> "NnueBatch":
        """把批次中的全部张量移动到指定设备。

        参数:
            device: PyTorch 设备对象或 ``cpu``、``cuda``、``mps`` 等设备名。

        返回:
            张量位于目标设备的新批次；原批次保持不变。
        """

        return NnueBatch(
            red_indices=self.red_indices.to(device),
            red_offsets=self.red_offsets.to(device),
            black_indices=self.black_indices.to(device),
            black_offsets=self.black_offsets.to(device),
            red_to_move=self.red_to_move.to(device),
            targets=self.targets.to(device),
            weights=self.weights.to(device),
        )


def pack_feature_lists(feature_lists: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    """把变长特征列表打包为扁平编号和分段边界。

    参数:
        feature_lists: 按样本排列的激活 HalfKA 特征编号。

    返回:
        ``(indices, offsets)``；两者均为 ``torch.long`` CPU 张量。
    """

    flat: list[int] = []
    offsets = [0]
    for features in feature_lists:
        flat.extend(features)
        offsets.append(len(flat))
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


class HalfKANetwork(nn.Module):
    """参数形状和排列顺序与 C++ ``NnueNetwork`` 完全一致的训练网络。

    网络结构为共享特征变换 ``11340→256``、双视角拼接 ``512``、隐藏层
    ``512→32`` 和输出层 ``32→1``。两处隐藏输出都使用 ``clamp(0, 1)``。
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_weights = nn.Parameter(
            torch.empty(FEATURE_DIMENSIONS, ACCUMULATOR_SIZE)
        )
        self.feature_bias = nn.Parameter(torch.zeros(ACCUMULATOR_SIZE))
        self.hidden = nn.Linear(CONCATENATED_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化稀疏特征变换和两个全连接层的可训练参数。"""

        nn.init.normal_(self.feature_weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.feature_bias)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def _transform(self, indices: Tensor, offsets: Tensor) -> Tensor:
        """计算每个样本的第一层未激活累加器。

        参数:
            indices: 整批扁平的激活特征编号。
            offsets: 每个样本在 ``indices`` 中的起止边界。

        返回:
            形状为 ``[batch, 256]`` 的 ``bias + Σ feature_weight``。

        异常:
            TypeError: 编号或边界不是 ``torch.long``。
            ValueError: 边界数量、顺序或末尾位置与编号数组不匹配。
        """

        if indices.dtype != torch.long or offsets.dtype != torch.long:
            raise TypeError("feature indices and offsets must use torch.long")
        if offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError("offsets must contain one start plus one end per sample")

        batch_size = offsets.numel() - 1
        lengths = offsets[1:] - offsets[:-1]
        if torch.any(lengths < 0) or int(offsets[-1].item()) != indices.numel():
            raise ValueError("feature offsets do not match packed indices")

        # expand 不分配每行存储；clone 后才能安全地用 index_add_ 原地累加。
        result = self.feature_bias.unsqueeze(0).expand(batch_size, -1).clone()
        if indices.numel() == 0:
            return result

        # 为每个激活特征生成所属样本行号，再将对应权重行累加到该样本。
        rows = torch.repeat_interleave(
            torch.arange(batch_size, device=indices.device), lengths
        )
        result.index_add_(0, rows, self.feature_weights.index_select(0, indices))
        return result

    def accumulators(
        self,
        red_indices: Tensor,
        red_offsets: Tensor,
        black_indices: Tensor,
        black_offsets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """计算红黑双方未激活的特征变换结果。

        参数:
            red_indices: 红方视角扁平特征编号。
            red_offsets: 红方视角样本边界。
            black_indices: 黑方视角扁平特征编号。
            black_offsets: 黑方视角样本边界。

        返回:
            ``(red_accumulator, black_accumulator)``，形状均为 ``[batch, 256]``。
        """

        return (
            self._transform(red_indices, red_offsets),
            self._transform(black_indices, black_offsets),
        )

    def forward(
        self,
        red_indices: Tensor,
        red_offsets: Tensor,
        black_indices: Tensor,
        black_offsets: Tensor,
        red_to_move: Tensor,
    ) -> Tensor:
        """从当前行棋方视角评估一个稀疏批次。

        参数:
            red_indices: 红方视角扁平特征编号。
            red_offsets: 红方视角样本边界。
            black_indices: 黑方视角扁平特征编号。
            black_offsets: 黑方视角样本边界。
            red_to_move: 每个样本是否红方行棋，形状为 ``[batch]``。

        返回:
            形状为 ``[batch]`` 的未取整 float32 引擎分值。

        异常:
            ValueError: 行棋方张量形状与批大小不匹配。
        """

        red, black = self.accumulators(
            red_indices, red_offsets, black_indices, black_offsets
        )
        red = torch.clamp(red, 0.0, 1.0)
        black = torch.clamp(black, 0.0, 1.0)

        if red_to_move.ndim != 1 or red_to_move.numel() != red.shape[0]:
            raise ValueError("red_to_move must contain one value per sample")
        # C++ 约定把当前方累加器放前 256 维，对方累加器放后 256 维。
        red_first = torch.cat((red, black), dim=1)
        black_first = torch.cat((black, red), dim=1)
        combined = torch.where(red_to_move[:, None], red_first, black_first)
        hidden = torch.clamp(self.hidden(combined), 0.0, 1.0)
        return self.output(hidden).squeeze(1)

    def forward_batch(self, batch: NnueBatch) -> Tensor:
        """使用 ``NnueBatch`` 调用前向传播。

        参数:
            batch: ``collate_samples`` 产生的训练批次。

        返回:
            形状为 ``[batch]`` 的预测分数。
        """

        return self(
            batch.red_indices,
            batch.red_offsets,
            batch.black_indices,
            batch.black_offsets,
            batch.red_to_move,
        )
