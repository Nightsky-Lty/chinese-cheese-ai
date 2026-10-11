"""定义与 C++ ``NnueNetwork`` 参数布局一致的 PyTorch 网络。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .halfka import FEATURE_DIMENSIONS

ACCUMULATOR_SIZE = 256
CONCATENATED_SIZE = ACCUMULATOR_SIZE * 2
HIDDEN_SIZE = 32
EVALUATION_LIMIT = 28_000
ARCHITECTURE_NAME = "halfka-11340-256x2-32-1-f32"
DEFAULT_SCORE_SCALE = 600.0
MAX_LAYER_WIDTH = 2048
MAX_HIDDEN_LAYERS = 8


@dataclass(frozen=True)
class NetworkConfig:
    """网络尺寸；输入保持 HalfKA，输出保持一个分数，所有中间层可配置。"""

    accumulator_size: int = ACCUMULATOR_SIZE
    hidden_sizes: tuple[int, ...] = (HIDDEN_SIZE,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        widths = (self.accumulator_size, *self.hidden_sizes)
        if not 1 <= len(self.hidden_sizes) <= MAX_HIDDEN_LAYERS:
            raise ValueError("hidden layer count must be between 1 and 8")
        if any(type(width) is not int or not 1 <= width <= MAX_LAYER_WIDTH for width in widths):
            raise ValueError("layer widths must be integers between 1 and 2048")

    @property
    def name(self) -> str:
        """生成包含全部尺寸的实验名称，避免不同架构混淆。"""
        hidden = "-".join(map(str, self.hidden_sizes))
        return f"halfka-{FEATURE_DIMENSIONS}-{self.accumulator_size}x2-{hidden}-1-f32"


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
        results: 当前行棋方视角的赛果，胜/和/负分别为 1/0/-1。
        result_known: 对应赛果是否真实存在；未知赛果绝不当作和棋。
    """

    red_indices: Tensor
    red_offsets: Tensor
    black_indices: Tensor
    black_offsets: Tensor
    red_to_move: Tensor
    targets: Tensor
    weights: Tensor
    results: Tensor
    result_known: Tensor
    phases: list[str] | None = None

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
            results=self.results.to(device),
            result_known=self.result_known.to(device),
            phases=self.phases,
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

    默认结构为 ``11340→256×2→32→1``，累加器宽度及各隐藏层宽度可配置。
    特征变换及每个隐藏层使用 ``clamp(0, 1)``，输出层不截断。
    """

    def __init__(self, score_scale: float = 1.0, *,
                 accumulator_size: int = ACCUMULATOR_SIZE,
                 hidden_sizes: Sequence[int] = (HIDDEN_SIZE,)) -> None:
        """构造网络。

        参数:
            score_scale: 原始输出到引擎分数的倍数，正式训练默认使用 600。
            accumulator_size: 每个视角共享特征变换的宽度，双视角拼接后为两倍。
            hidden_sizes: 从前到后每个隐藏层的宽度；默认仅一个 32 维隐藏层。
        """
        super().__init__()
        if not math.isfinite(score_scale) or score_scale <= 0.0:
            raise ValueError("score scale must be finite and positive")
        # 不属于 state_dict；检查点的 scoring 元数据决定末层输出单位。
        self.score_scale = float(score_scale)
        self.config = NetworkConfig(accumulator_size, tuple(hidden_sizes))
        self.feature_weights = nn.Parameter(
            torch.empty(FEATURE_DIMENSIONS, accumulator_size)
        )
        self.feature_bias = nn.Parameter(torch.zeros(accumulator_size))
        # 保留第一层的 hidden.* 参数名，使现有模型无需改写即可加载。
        self.hidden = nn.Linear(accumulator_size * 2, self.config.hidden_sizes[0])
        self.extra_hidden = nn.ModuleList(
            nn.Linear(previous, following)
            for previous, following in zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        )
        self.output = nn.Linear(self.config.hidden_sizes[-1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化稀疏特征变换、全部隐藏层和线性输出层的可训练参数。"""

        nn.init.normal_(self.feature_weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.feature_bias)
        for layer in self.dense_layers():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def dense_layers(self) -> tuple[nn.Linear, ...]:
        """按推理顺序返回全部隐藏层，不包含最终线性输出层。"""
        return (self.hidden, *self.extra_hidden)

    def _transform(self, indices: Tensor, offsets: Tensor) -> Tensor:
        """计算每个样本的第一层未激活累加器。

        参数:
            indices: 整批扁平的激活特征编号。
            offsets: 每个样本在 ``indices`` 中的起止边界。

        返回:
            形状为 ``[batch, accumulator_size]`` 的 ``bias + Σ feature_weight``。

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
            ``(red_accumulator, black_accumulator)``，宽度为配置的累加器尺寸。
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
        """从当前行棋方视角评估一个稀疏批次。"""

        return self.forward_with_activations(
            red_indices,
            red_offsets,
            black_indices,
            black_offsets,
            red_to_move,
        )[0]

    def forward_with_activations(
        self,
        red_indices: Tensor,
        red_offsets: Tensor,
        black_indices: Tensor,
        black_offsets: Tensor,
        red_to_move: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """从当前行棋方视角评估一个稀疏批次。

        参数:
            red_indices: 红方视角扁平特征编号。
            red_offsets: 红方视角样本边界。
            black_indices: 黑方视角扁平特征编号。
            black_offsets: 黑方视角样本边界。
            red_to_move: 每个样本是否红方行棋，形状为 ``[batch]``。

        返回:
            ``(raw_output, accumulator_pre_activation, hidden_pre_activation)``。
            输出形状为 ``[batch]``；乘以 score_scale 得引擎分值。后两项用于
            测量 clipped-ReLU 的上下饱和比例，不改变网络计算。
            多隐藏层的激活前值按维度拼接，诊断按全部隐藏神经元汇总。

        异常:
            ValueError: 行棋方张量形状与批大小不匹配。
        """

        red, black = self.accumulators(
            red_indices, red_offsets, black_indices, black_offsets
        )
        if red_to_move.ndim != 1 or red_to_move.numel() != red.shape[0]:
            raise ValueError("red_to_move must contain one value per sample")
        accumulator_pre_activation = torch.cat((red, black), dim=0)
        red = torch.clamp(red, 0.0, 1.0)
        black = torch.clamp(black, 0.0, 1.0)
        # C++ 约定当前方累加器在前，对方累加器在后。
        red_first = torch.cat((red, black), dim=1)
        black_first = torch.cat((black, red), dim=1)
        combined = torch.where(red_to_move[:, None], red_first, black_first)
        hidden = combined
        pre_activations = []
        for layer in self.dense_layers():
            values = layer(hidden)
            pre_activations.append(values)
            hidden = torch.clamp(values, 0.0, 1.0)
        # 多隐藏层诊断按神经元汇总；保留现有诊断接口的三元组约定。
        hidden_pre_activation = torch.cat(pre_activations, dim=1)
        return (
            self.output(hidden).squeeze(1),
            accumulator_pre_activation,
            hidden_pre_activation,
        )

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

    def score_batch(self, batch: NnueBatch) -> Tensor:
        """返回当前行棋方视角、以引擎分数为单位的批量评分。"""

        return self.forward_batch(batch) * self.score_scale
