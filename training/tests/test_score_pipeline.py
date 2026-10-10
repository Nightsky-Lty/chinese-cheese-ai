"""端到端检查评分单位迁移、推理导出与训练诊断。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.checkpoint import (
    LEGACY_CHECKPOINT_FORMAT,
    architecture_metadata,
    load_model,
    migrate_output_score_scale,
    save_checkpoint,
)
from training.dataset import TrainingSample, collate_samples
from training.diagnostics import EpochMetrics
from training.export_nnue import HEADER, export_checkpoint, export_model
from training.halfka import encode_fen
from training.model import ACCUMULATOR_SIZE, HIDDEN_SIZE, HalfKANetwork
from training.train import validate_resume_split, weighted_huber_loss
from training.verify_cpp import compare_scores, cpp_score, python_score

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
ASYMMETRIC_FEN = "4k4/9/9/9/4p4/9/9/4R4/9/4K4 b"
ENGINE = Path(__file__).resolve().parents[2] / "build/make/xiangqi_cli"


class ScoreScaleTests(unittest.TestCase):
    def test_python_cpp_tolerance_accepts_float_drift_but_rejects_one_point(self) -> None:
        # This is the observed scale of f32 accumulation-order noise in C++.
        tolerated_drift = compare_scores(72.125, 72.125183095)
        self.assertTrue(tolerated_drift["matches"])
        self.assertAlmostEqual(tolerated_drift["allowed_limit"], 0.001072125)

        self.assertTrue(compare_scores(0.08, 0.080183095)["matches"])
        self.assertFalse(compare_scores(0.08, 1.08)["matches"])
        self.assertFalse(compare_scores(28000.0, 28001.0)["matches"])

    def test_python_cpp_tolerance_rejects_non_finite_scores(self) -> None:
        self.assertFalse(compare_scores(float("nan"), 0.0)["matches"])
        self.assertFalse(compare_scores(0.0, float("inf"))["matches"])
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            compare_scores(0.0, 0.0, absolute_tolerance=float("nan"))

    def test_huber_target_and_beta_share_normalized_units(self) -> None:
        samples = [
            TrainingSample(encode_fen(INITIAL_FEN), 120.0, weight=2.0),
            TrainingSample(encode_fen(ASYMMETRIC_FEN), -900.0),
        ]
        batch = collate_samples(samples)
        normalized_predictions = torch.tensor([0.4, -1.0])
        scale = 600.0
        beta_engine = 120.0

        actual = weighted_huber_loss(
            normalized_predictions, batch, beta_engine, score_scale=scale
        )
        expected_items = F.smooth_l1_loss(
            normalized_predictions,
            batch.targets / scale,
            reduction="none",
            beta=beta_engine / scale,
        )
        expected = (expected_items * batch.weights).sum() / batch.weights.sum()
        torch.testing.assert_close(actual, expected)

    def test_legacy_checkpoint_migrates_once_and_preserves_engine_score(self) -> None:
        torch.manual_seed(41)
        model = HalfKANetwork(score_scale=1.0)
        with torch.no_grad():
            model.output.weight.normal_(mean=0.0, std=0.1)
            model.output.bias.fill_(8.25)

        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy-v1.pt"
            torch.save(
                {
                    "format": LEGACY_CHECKPOINT_FORMAT,
                    "architecture": architecture_metadata(),
                    "model_state_dict": model.state_dict(),
                },
                legacy_path,
            )
            loaded = load_model(legacy_path)
            original_score = python_score(loaded, ASYMMETRIC_FEN)
            migrate_output_score_scale(loaded, 1.0, 600.0)
            migrated_score = python_score(loaded, ASYMMETRIC_FEN)
            self.assertEqual(loaded.score_scale, 600.0)
            self.assertAlmostEqual(migrated_score, original_score, delta=1.0e-4)

            normalized_path = Path(directory) / "normalized.pt"
            save_checkpoint(
                normalized_path,
                {"model_state_dict": loaded.state_dict()},
                score_scale=loaded.score_scale,
            )
            round_tripped = load_model(normalized_path)
            self.assertAlmostEqual(
                python_score(round_tripped, ASYMMETRIC_FEN), original_score,
                delta=1.0e-4,
            )

    def test_export_restores_engine_units_in_final_layer(self) -> None:
        torch.manual_seed(8)
        engine_model = HalfKANetwork(score_scale=1.0)
        with torch.no_grad():
            engine_model.output.weight.normal_(mean=0.0, std=0.02)
            engine_model.output.bias.fill_(-3.75)
        expected_score = python_score(engine_model, ASYMMETRIC_FEN)

        normalized_model = HalfKANetwork(score_scale=600.0)
        normalized_model.load_state_dict(engine_model.state_dict())
        migrate_output_score_scale(normalized_model, 1.0, 600.0)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "export.nnue"
            export_model(normalized_model, output_path)
            payload = output_path.read_bytes()
            prefix_floats = (
                ACCUMULATOR_SIZE
                + normalized_model.feature_weights.numel()
                + HIDDEN_SIZE
                + normalized_model.hidden.weight.numel()
            )
            last_layer_start = HEADER.size + prefix_floats * 4
            final_layer = np.frombuffer(
                payload,
                dtype="<f4",
                count=HIDDEN_SIZE + 1,
                offset=last_layer_start,
            )
            expected_final_layer = torch.cat(
                (
                    engine_model.output.bias.detach(),
                    engine_model.output.weight.detach().reshape(-1),
                )
            ).numpy()
            np.testing.assert_allclose(final_layer, expected_final_layer, rtol=1e-6)

            checkpoint_path = Path(directory) / "normalized.pt"
            save_checkpoint(
                checkpoint_path,
                {"model_state_dict": normalized_model.state_dict()},
                score_scale=600.0,
            )
            exported_path = Path(directory) / "checkpoint.nnue"
            export_checkpoint(checkpoint_path, exported_path)
            self.assertAlmostEqual(
                python_score(normalized_model, ASYMMETRIC_FEN),
                expected_score,
                delta=1.0e-4,
            )

    def test_python_cpp_raw_scores_match_when_cli_is_available(self) -> None:
        if not ENGINE.is_file():
            self.skipTest("C++ CLI is not built")
        torch.manual_seed(17)
        model = HalfKANetwork(score_scale=600.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "model.pt"
            nnue_path = root / "model.nnue"
            save_checkpoint(
                checkpoint_path,
                {"model_state_dict": model.state_dict()},
                score_scale=600.0,
            )
            export_checkpoint(checkpoint_path, nnue_path)
            for fen in (INITIAL_FEN, ASYMMETRIC_FEN):
                expected = python_score(model, fen)
                actual = cpp_score(ENGINE, nnue_path, fen)
                comparison = compare_scores(expected, actual)
                self.assertTrue(
                    comparison["matches"],
                    msg=f"Python/C++ drift exceeded {comparison['allowed_limit']}: "
                    f"{expected} vs {actual}",
                )


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_resume_requires_explicit_reset_when_split_changes(self) -> None:
        current = {"sample_digests": {"train": "new"}}
        checkpoint = {"split_metadata": {"sample_digests": {"train": "old"}}}
        with self.assertRaisesRegex(ValueError, "different data split"):
            validate_resume_split(checkpoint, current, reset_best=False)
        compatible, reason = validate_resume_split(
            checkpoint, current, reset_best=True
        )
        self.assertFalse(compatible)
        self.assertEqual(reason, "data split changed")
        self.assertEqual(
            validate_resume_split({"epoch": 4}, current, reset_best=False),
            (False, "checkpoint has no split compatibility metadata"),
        )

    def test_metrics_include_components_saturation_phases_and_unknown_results(self) -> None:
        samples = [
            TrainingSample(
                encode_fen(INITIAL_FEN), 0.0, result=1, phase="opening"
            ),
            TrainingSample(
                encode_fen(ASYMMETRIC_FEN), 400.0, result=0, phase="middlegame"
            ),
            TrainingSample(
                encode_fen(INITIAL_FEN[:-1] + "b"), 1000.0, phase="endgame"
            ),
        ]
        batch = collate_samples(samples)
        model = HalfKANetwork(score_scale=600.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias.fill_(0.5)
        predictions, accumulator_pre, hidden_pre = model.forward_with_activations(
            batch.red_indices,
            batch.red_offsets,
            batch.black_indices,
            batch.black_offsets,
            batch.red_to_move,
        )

        metrics = EpochMetrics(
            score_scale=600.0,
            huber_beta=100.0,
            outcome_scale=600.0,
            outcome_weight=0.0,
        )
        metrics.update(predictions, batch, accumulator_pre, hidden_pre)
        report = metrics.finish()

        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["result_known_count"], 2)
        self.assertEqual(report["result_unknown_count"], 1)
        self.assertEqual(report["result_counts"], {"loss": 0, "draw": 1, "win": 1})
        self.assertAlmostEqual(report["mae_engine"], (300.0 + 100.0 + 700.0) / 3)
        self.assertEqual(report["prediction_quantiles_engine"]["p50"], 300.0)
        self.assertEqual(
            {name: values["sample_count"] for name, values in report["piece_phase_errors"].items()},
            {"opening": 1, "middlegame": 1, "endgame": 1},
        )
        self.assertEqual(
            report["activation_saturation"]["accumulator"]["low_fraction"], 1.0
        )
        self.assertEqual(
            report["activation_saturation"]["hidden"]["low_fraction"], 1.0
        )
        self.assertTrue(np.isfinite(report["score_loss"]))
        self.assertTrue(np.isfinite(report["outcome_loss"]))
        self.assertEqual(report["weighted_outcome_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
