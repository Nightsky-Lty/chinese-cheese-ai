"""训练指标累计与持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .model import NnueBatch

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


class EpochMetrics:
    """聚合一轮模型输出、损失、饱和率和按棋子数划分的误差。"""

    def __init__(
        self,
        *,
        score_scale: float,
        huber_beta: float,
        outcome_scale: float,
        outcome_weight: float,
    ) -> None:
        self.score_scale = score_scale
        self.huber_beta = huber_beta
        self.outcome_scale = outcome_scale
        self.outcome_weight = outcome_weight
        self.predictions: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.results: list[np.ndarray] = []
        self.result_known: list[np.ndarray] = []
        self.piece_counts: list[np.ndarray] = []
        self.phases: list[np.ndarray] = []
        self.score_losses: list[np.ndarray] = []
        self.outcome_losses: list[np.ndarray] = []
        self.total_weight = 0.0
        self.accumulator_values = 0
        self.accumulator_low = 0
        self.accumulator_high = 0
        self.hidden_values = 0
        self.hidden_low = 0
        self.hidden_high = 0

    @staticmethod
    def _cpu(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(device="cpu", dtype=torch.float64).numpy()

    def update(
        self,
        predictions: torch.Tensor,
        batch: NnueBatch,
        accumulator_pre_activation: torch.Tensor,
        hidden_pre_activation: torch.Tensor,
    ) -> None:
        """把一个批次计入本轮指标。"""

        normalized_targets = batch.targets / self.score_scale
        score_losses = F.smooth_l1_loss(
            predictions,
            normalized_targets,
            reduction="none",
            beta=self.huber_beta / self.score_scale,
        )
        known = batch.result_known
        outcome_losses = torch.zeros_like(predictions)
        if bool(torch.any(known)):
            outcome_targets = (batch.results[known] + 1.0) / 2.0
            outcome_losses[known] = F.binary_cross_entropy_with_logits(
                predictions[known] * (self.score_scale / self.outcome_scale),
                outcome_targets,
                reduction="none",
            )

        self.predictions.append(self._cpu(predictions * self.score_scale))
        self.targets.append(self._cpu(batch.targets))
        self.weights.append(self._cpu(batch.weights))
        self.results.append(self._cpu(batch.results))
        self.result_known.append(self._cpu(known))
        piece_counts = self._cpu(batch.red_offsets[1:] - batch.red_offsets[:-1])
        self.piece_counts.append(piece_counts)
        if batch.phases is None:
            self.phases.append(
                np.where(
                    piece_counts >= 24,
                    "opening",
                    np.where(piece_counts >= 14, "middlegame", "endgame"),
                )
            )
        else:
            self.phases.append(np.asarray(batch.phases, dtype=str))
        self.score_losses.append(self._cpu(score_losses))
        self.outcome_losses.append(self._cpu(outcome_losses))
        self.total_weight += float(batch.weights.sum().detach().cpu())

        accumulator = accumulator_pre_activation.detach()
        hidden = hidden_pre_activation.detach()
        self.accumulator_values += accumulator.numel()
        self.accumulator_low += int(torch.count_nonzero(accumulator <= 0.0).cpu())
        self.accumulator_high += int(torch.count_nonzero(accumulator >= 1.0).cpu())
        self.hidden_values += hidden.numel()
        self.hidden_low += int(torch.count_nonzero(hidden <= 0.0).cpu())
        self.hidden_high += int(torch.count_nonzero(hidden >= 1.0).cpu())

    @staticmethod
    def _quantiles(values: np.ndarray) -> dict[str, float]:
        names = ("min", "p05", "p25", "p50", "p75", "p95", "max")
        return {
            name: float(value)
            for name, value in zip(names, np.quantile(values, QUANTILES))
        }

    def finish(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的指标字典。"""

        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        weights = np.concatenate(self.weights)
        results = np.concatenate(self.results)
        known = np.concatenate(self.result_known).astype(bool)
        pieces = np.concatenate(self.piece_counts).astype(np.int64)
        score_losses = np.concatenate(self.score_losses)
        outcome_losses = np.concatenate(self.outcome_losses)

        score_loss = float(np.dot(score_losses, weights) / self.total_weight)
        outcome_loss = (
            float(np.dot(outcome_losses[known], weights[known]) / self.total_weight)
            if np.any(known)
            else 0.0
        )
        absolute_error = np.abs(predictions - targets)
        weighted_mae = float(np.dot(absolute_error, weights) / self.total_weight)

        phase_labels = np.concatenate(self.phases)
        phases = {
            phase: phase_labels == phase
            for phase in ("opening", "middlegame", "endgame")
        }
        phase_metrics: dict[str, dict[str, float | int]] = {}
        for phase, mask in phases.items():
            if not np.any(mask):
                phase_metrics[phase] = {"sample_count": 0, "mae_engine": 0.0}
                continue
            errors = absolute_error[mask]
            phase_metrics[phase] = {
                "sample_count": int(np.count_nonzero(mask)),
                "mae_engine": float(errors.mean()),
                "weighted_mae_engine": float(
                    np.dot(errors, weights[mask]) / weights[mask].sum()
                ),
            }

        sample_count = int(predictions.size)
        return {
            "sample_count": sample_count,
            "result_known_count": int(np.count_nonzero(known)),
            "result_unknown_count": int(sample_count - np.count_nonzero(known)),
            "score_loss": score_loss,
            "outcome_loss": outcome_loss,
            "weighted_outcome_loss": self.outcome_weight * outcome_loss,
            "total_loss": score_loss + self.outcome_weight * outcome_loss,
            "mae_engine": float(absolute_error.mean()),
            "weighted_mae_engine": weighted_mae,
            "prediction_quantiles_engine": self._quantiles(predictions),
            "label_quantiles_engine": self._quantiles(targets),
            "result_counts": {
                "loss": int(np.count_nonzero(known & (results < 0.0))),
                "draw": int(np.count_nonzero(known & (results == 0.0))),
                "win": int(np.count_nonzero(known & (results > 0.0))),
            },
            "piece_phase_errors": phase_metrics,
            "activation_saturation": {
                "accumulator": {
                    "low_fraction": self.accumulator_low / self.accumulator_values,
                    "high_fraction": self.accumulator_high / self.accumulator_values,
                    "value_count": self.accumulator_values,
                },
                "hidden": {
                    "low_fraction": self.hidden_low / self.hidden_values,
                    "high_fraction": self.hidden_high / self.hidden_values,
                    "value_count": self.hidden_values,
                },
            },
        }


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    """持久追加一条 JSON Lines 记录。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    """原子写入 UTF-8 JSON 文件。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
