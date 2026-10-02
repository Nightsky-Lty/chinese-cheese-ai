"""验证 Python 特征、训练反传、权重导出和 C++ 数值一致性。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import load_model, save_checkpoint
from training.dataset import JsonlPositionDataset, TrainingSample, collate_samples
from training.export_nnue import HEADER, export_checkpoint
from training.halfka import FEATURE_DIMENSIONS, encode_fen
from training.model import (
    ACCUMULATOR_SIZE,
    HIDDEN_SIZE,
    HalfKANetwork,
)
from training.verify_cpp import cpp_score, python_score

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
ASYMMETRIC_FEN = "4k4/9/9/9/4p4/9/9/4R4/9/4K4 w"


class HalfKAEncodingTests(unittest.TestCase):
    def test_initial_position_is_sparse_and_symmetric(self) -> None:
        encoded = encode_fen(INITIAL_FEN)
        self.assertTrue(encoded.red_to_move)
        self.assertEqual(len(encoded.red_features), 32)
        self.assertEqual(encoded.red_features, encoded.black_features)
        self.assertEqual(len(set(encoded.red_features)), 32)
        self.assertTrue(all(0 <= feature < FEATURE_DIMENSIONS for feature in encoded.red_features))

    def test_side_to_move_and_invalid_fen(self) -> None:
        self.assertFalse(encode_fen(INITIAL_FEN[:-1] + "b").red_to_move)
        with self.assertRaisesRegex(ValueError, "exactly one king"):
            encode_fen("9/9/9/9/9/9/9/9/9/9 w")
        with self.assertRaisesRegex(ValueError, "palace"):
            encode_fen("k8/9/9/9/9/9/9/9/9/K8 w")


class DatasetAndModelTests(unittest.TestCase):
    def test_jsonl_dataset_and_sparse_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            records = [
                {"fen": INITIAL_FEN, "score": 0},
                {"fen": ASYMMETRIC_FEN, "score": 500, "weight": 2.0},
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            dataset = JsonlPositionDataset(path)
            batch = collate_samples([dataset[0], dataset[1]])

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(batch.red_offsets.shape), (3,))
        self.assertEqual(int(batch.red_offsets[-1]), 36)
        self.assertEqual(batch.targets.tolist(), [0.0, 500.0])
        self.assertEqual(batch.weights.tolist(), [1.0, 2.0])

    def test_forward_and_backward(self) -> None:
        model = HalfKANetwork()
        samples = [
            TrainingSample(encode_fen(INITIAL_FEN), 0.0),
            TrainingSample(encode_fen(ASYMMETRIC_FEN), 500.0),
        ]
        batch = collate_samples(samples)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
        predictions = model.forward_batch(batch)
        loss = torch.nn.functional.smooth_l1_loss(predictions, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.feature_weights.grad)

    def test_current_side_controls_accumulator_order(self) -> None:
        model = HalfKANetwork()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            encoded = encode_fen(ASYMMETRIC_FEN)
            red_only_feature = next(
                feature
                for feature in encoded.red_features
                if feature not in set(encoded.black_features)
            )
            model.feature_weights[red_only_feature, 0] = 0.5
            model.hidden.weight[0, 0] = 1.0
            model.output.weight[0, 0] = 100.0

        red_sample = TrainingSample(encoded, 0.0)
        black_position = encode_fen(ASYMMETRIC_FEN[:-1] + "b")
        black_sample = TrainingSample(black_position, 0.0)
        scores = model.forward_batch(collate_samples([red_sample, black_sample]))
        self.assertGreater(float(scores[0].detach()), float(scores[1].detach()))


class CheckpointAndExportTests(unittest.TestCase):
    def _make_model(self) -> HalfKANetwork:
        torch.manual_seed(7)
        model = HalfKANetwork()
        with torch.no_grad():
            model.feature_weights.zero_()
            model.feature_bias.copy_(torch.linspace(-0.2, 0.3, ACCUMULATOR_SIZE))
            active = set()
            for fen in (INITIAL_FEN, ASYMMETRIC_FEN):
                encoded = encode_fen(fen)
                active.update(encoded.red_features)
                active.update(encoded.black_features)
            row = torch.linspace(-0.03, 0.04, ACCUMULATOR_SIZE)
            for index, feature in enumerate(sorted(active)):
                model.feature_weights[feature].copy_(row * ((index % 5) + 1))
            model.hidden.weight.copy_(
                torch.linspace(
                    -0.01,
                    0.01,
                    HIDDEN_SIZE * ACCUMULATOR_SIZE * 2,
                ).reshape(HIDDEN_SIZE, ACCUMULATOR_SIZE * 2)
            )
            model.hidden.bias.copy_(torch.linspace(-0.1, 0.1, HIDDEN_SIZE))
            model.output.weight.copy_(
                torch.linspace(-2.0, 3.0, HIDDEN_SIZE).reshape(1, HIDDEN_SIZE)
            )
            model.output.bias.fill_(17.25)
        return model

    def test_checkpoint_export_and_cpp_parity(self) -> None:
        engine = Path("build/make/xiangqi_cli")
        if not engine.is_file():
            self.skipTest("C++ CLI is not built")

        model = self._make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            nnue_path = Path(directory) / "model.nnue"
            save_checkpoint(
                checkpoint_path,
                {"model_state_dict": model.state_dict(), "epoch": 0},
            )
            loaded = load_model(checkpoint_path)
            export_checkpoint(checkpoint_path, nnue_path)

            float_count = (
                ACCUMULATOR_SIZE
                + FEATURE_DIMENSIONS * ACCUMULATOR_SIZE
                + HIDDEN_SIZE
                + HIDDEN_SIZE * ACCUMULATOR_SIZE * 2
                + 1
                + HIDDEN_SIZE
            )
            self.assertEqual(nnue_path.stat().st_size, HEADER.size + float_count * 4)
            for fen in (INITIAL_FEN, ASYMMETRIC_FEN):
                expected = python_score(loaded, fen)
                actual = cpp_score(engine, nnue_path, fen)
                self.assertAlmostEqual(actual, expected, delta=1.0e-4)


if __name__ == "__main__":
    unittest.main()
