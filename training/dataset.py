"""加载 JSONL 监督样本并组成 HalfKA 稀疏训练批次。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .halfka import EncodedPosition, encode_fen
from .model import NnueBatch, pack_feature_lists
from .partition import (
    canonical_position_key,
    file_fingerprint,
    game_group_key,
    source_id,
)

PHASES = {"opening", "middlegame", "endgame"}


def phase_for_fen(fen: str) -> str:
    """按引擎的非将棋子物质阈值推导缺失的局面阶段。"""

    values = {"r": 900, "n": 400, "b": 200, "a": 200, "c": 450, "p": 100}
    board = fen.split()[0]
    material = sum(values.get(piece.lower(), 0) for piece in board if piece.isalpha())
    if material * 100 >= 9600 * 80:
        return "opening"
    if material * 100 >= 9600 * 35:
        return "middlegame"
    return "endgame"


@dataclass(frozen=True)
class TrainingSample:
    """一个已编码局面及其监督信息。

    属性:
        position: 红黑双视角 HalfKA 特征和当前行棋方。
        target: 当前行棋方视角的目标引擎分值。
        weight: 该样本在损失函数中的正数权重。
        fen: 原始 FEN，主要用于诊断和复现问题。
        result: 当前行棋方视角的赛果；None 表示对局未得出赛果。
        group_key: 由数据文件和对局 ID 组成的稳定分组键。
        position_key: 局面及其水平镜像的共同键，用于跨区去重。
        phase: 局面阶段；旧样本缺失时按引擎的非将物质值推导。
        source_fingerprint: 原始数据文件的 SHA-256。
    """

    position: EncodedPosition
    target: float
    weight: float = 1.0
    fen: str = ""
    result: int | None = None
    group_key: str = ""
    position_key: str = ""
    phase: str | None = None
    source_fingerprint: str = ""


class JsonlPositionDataset(Dataset[TrainingSample]):
    """把 JSON Lines 文件加载到内存并预编码所有特征。

    每个非空、非注释行必须包含 ``fen`` 和 ``score``。``score`` 采用当前行棋方
    视角，并与 C++ 评估器使用同一分值单位。可选的正数 ``weight`` 调整该样本
    的损失权重。可选的 ``result`` 为当前行棋方视角的 -1、0 或 1；省略或
    ``null`` 表示未知，不能解释为和棋。第一版使用内存数据集便于验证；超大数据集
    后续应改为分片流式加载。
    """

    def __init__(
        self, path: str | Path, score_clip: float = 28_000.0,
        quarantine_sources: set[str] | None = None,
        narrow_score_limit: float = 2.0,
        allow_empty: bool = False,
        group_filter: set[str] | None = None,
    ) -> None:
        """读取、校验并编码一个 JSONL 数据集。

        参数:
            path: JSON Lines 训练文件路径。
            score_clip: 标签绝对值上限，默认避开 C++ 将杀分数区间。
            quarantine_sources: 需要隔离窄评分的旧文件标识集合。
            narrow_score_limit: 被隔离文件中要舍弃的评分绝对值上限。
            allow_empty: 若整份旧数据都被隔离，允许返回空数据集供上层跳过。
            group_filter: 可选的对局组白名单，用于只编码固定测试锚点。

        异常:
            OSError: 文件无法读取。
            ValueError: 记录、标签、权重、FEN 非法，或文件没有有效样本。
        """

        self.path = Path(path)
        self.source_id = source_id(self.path)
        self.source_fingerprint = file_fingerprint(self.path)
        self.samples: list[TrainingSample] = []
        self.quarantined_count = 0
        self.maximum_absolute_score = 0.0
        quarantined_source = source_id(self.path) in (quarantine_sources or set())
        include_file_group = group_filter is not None and f"{self.source_id}#file" in group_filter
        saw_ungrouped = False
        saw_numbered_game = False
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    record: Any = json.loads(text)
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    group_key = game_group_key(self.path, record)
                    saw_ungrouped = saw_ungrouped or group_key.endswith("#ungrouped")
                    saw_numbered_game = saw_numbered_game or "#game=" in group_key
                    if (
                        group_filter is not None
                        and not include_file_group
                        and group_key not in group_filter
                    ):
                        continue
                    fen = str(record["fen"])
                    score = float(record["score"])
                    weight = float(record.get("weight", 1.0))
                    result = record.get("result")
                    phase = record.get("phase")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid training record at {self.path}:{line_number}"
                    ) from error
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score at {self.path}:{line_number}")
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"weight must be positive at {self.path}:{line_number}")
                if result is not None and (type(result) is not int or result not in (-1, 0, 1)):
                    raise ValueError(
                        f"result must be -1, 0, 1, or null at {self.path}:{line_number}"
                    )
                if phase is not None and (not isinstance(phase, str) or phase not in PHASES):
                    raise ValueError(
                        f"phase must be opening, middlegame, endgame, or null "
                        f"at {self.path}:{line_number}"
                    )
                if quarantined_source and abs(score) <= narrow_score_limit:
                    self.quarantined_count += 1
                    continue
                score = max(-score_clip, min(score_clip, score))
                self.maximum_absolute_score = max(self.maximum_absolute_score, abs(score))
                try:
                    encoded = encode_fen(fen)
                except ValueError as error:
                    raise ValueError(
                        f"invalid FEN at {self.path}:{line_number}: {error}"
                    ) from error
                self.samples.append(
                    TrainingSample(
                        encoded, score, weight, fen, result,
                        group_key, canonical_position_key(fen),
                        phase if phase is not None else phase_for_fen(fen),
                        self.source_fingerprint,
                    )
                )
        if saw_ungrouped and saw_numbered_game:
            # A missing game ID makes its relationship to numbered games
            # unknowable, so conservatively keep the whole source together.
            file_group = f"{self.source_id}#file"
            if group_filter is not None and file_group not in group_filter:
                self.samples.clear()
            else:
                self.samples = [
                    replace(sample, group_key=file_group) for sample in self.samples
                ]
        elif group_filter is not None and include_file_group:
            # A source-wide group filter only matches if the whole-file fallback
            # above was needed; otherwise records retain their individual IDs.
            self.samples.clear()
        if not self.samples and not allow_empty:
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
        包含扁平特征、分段边界、行棋方、评分、赛果掩码和权重的 ``NnueBatch``。

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
        results=torch.tensor(
            [sample.result if sample.result is not None else 0 for sample in samples],
            dtype=torch.float32,
        ),
        result_known=torch.tensor(
            [sample.result is not None for sample in samples], dtype=torch.bool
        ),
        phases=(
            [sample.phase for sample in samples]
            if all(sample.phase is not None for sample in samples)
            else None
        ),
    )
