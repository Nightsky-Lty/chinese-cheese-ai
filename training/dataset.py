"""加载 JSONL 监督样本并组成 HalfKA 稀疏训练批次。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .halfka import EncodedPosition, encode_fen
from .model import NnueBatch, pack_feature_lists


@dataclass(frozen=True)
class TrainingSample:
    """一个已编码局面及其监督信息。

    属性:
        position: 红黑双视角 HalfKA 特征和当前行棋方。
        target: 当前行棋方视角的目标引擎分值。
        weight: 该样本在损失函数中的正数权重。
        fen: 原始 FEN，主要用于诊断和复现问题。
    """

    position: EncodedPosition
    target: float
    weight: float = 1.0
    fen: str = ""


class JsonlPositionDataset(Dataset[TrainingSample]):
    """把 JSON Lines 文件加载到内存并预编码所有特征。

    每个非空、非注释行必须包含 ``fen`` 和 ``score``。``score`` 采用当前行棋方
    视角，并与 C++ 评估器使用同一分值单位。可选的正数 ``weight`` 调整该样本
    的损失权重。第一版使用内存数据集便于验证；超大数据集后续应改为分片流式加载。
    """

    def __init__(self, path: str | Path, score_clip: float = 28_000.0) -> None:
        """读取、校验并编码一个 JSONL 数据集。

        参数:
            path: JSON Lines 训练文件路径。
            score_clip: 标签绝对值上限，默认避开 C++ 将杀分数区间。

        异常:
            OSError: 文件无法读取。
            ValueError: 记录、标签、权重、FEN 非法，或文件没有有效样本。
        """

        self.path = Path(path)
        self.samples: list[TrainingSample] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    record: Any = json.loads(text)
                    fen = str(record["fen"])
                    score = float(record["score"])
                    weight = float(record.get("weight", 1.0))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid training record at {self.path}:{line_number}"
                    ) from error
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score at {self.path}:{line_number}")
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"weight must be positive at {self.path}:{line_number}")
                score = max(-score_clip, min(score_clip, score))
                try:
                    encoded = encode_fen(fen)
                except ValueError as error:
                    raise ValueError(
                        f"invalid FEN at {self.path}:{line_number}: {error}"
                    ) from error
                self.samples.append(TrainingSample(encoded, score, weight, fen))
        if not self.samples:
            raise ValueError(f"training dataset is empty: {self.path}")

    def __len__(self) -> int:
        """返回已加载的训练样本数量。"""

        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        """按下标返回一个预编码样本。

        参数:
            index: 范围为 ``[0, len(dataset))`` 的样本下标。

        返回:
            对应的 ``TrainingSample``。
        """

        return self.samples[index]


def collate_samples(samples: Sequence[TrainingSample]) -> NnueBatch:
    """把若干样本整理成不含 11340 维稠密向量的训练批次。

    参数:
        samples: 一个非空的 ``TrainingSample`` 序列。

    返回:
        包含扁平特征、分段边界、行棋方、标签和权重的 ``NnueBatch``。

    异常:
        ValueError: 输入样本序列为空。
    """

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    red_indices, red_offsets = pack_feature_lists(
        [sample.position.red_features for sample in samples]
    )
    black_indices, black_offsets = pack_feature_lists(
        [sample.position.black_features for sample in samples]
    )
    return NnueBatch(
        red_indices=red_indices,
        red_offsets=red_offsets,
        black_indices=black_indices,
        black_offsets=black_offsets,
        red_to_move=torch.tensor(
            [sample.position.red_to_move for sample in samples], dtype=torch.bool
        ),
        targets=torch.tensor([sample.target for sample in samples], dtype=torch.float32),
        weights=torch.tensor([sample.weight for sample in samples], dtype=torch.float32),
    )
