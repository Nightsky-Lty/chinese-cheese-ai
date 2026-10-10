"""验证 Python 特征、训练反传、权重导出和 C++ 数值一致性。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import (
    LEGACY_CHECKPOINT_FORMAT, architecture_metadata, load_model, save_checkpoint,
)
from training.arena import summarize_games
from training.dataset import JsonlPositionDataset, TrainingSample, collate_samples
from training.export_nnue import HEADER, export_checkpoint
from training.halfka import FEATURE_DIMENSIONS, encode_fen
from training.iterate import (
    STATE_FORMAT,
    atomic_write_json,
    load_state,
    replay_datasets,
)
from training.model import (
    ACCUMULATOR_SIZE,
    HIDDEN_SIZE,
    HalfKANetwork,
)
from training.partition import canonical_position_key, source_id, split_indices
from training.train import combined_loss, weighted_huber_loss
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
                {"fen": ASYMMETRIC_FEN, "score": 500, "weight": 2.0, "result": 1},
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
        self.assertEqual(batch.results.tolist(), [0.0, 1.0])
        self.assertEqual(batch.result_known.tolist(), [False, True])

    def test_unknown_result_is_masked_not_a_draw(self) -> None:
        samples = [
            TrainingSample(encode_fen(INITIAL_FEN), 0.0),
            TrainingSample(encode_fen(ASYMMETRIC_FEN), 0.0, result=1),
        ]
        batch = collate_samples(samples)
        predictions = torch.zeros(2, requires_grad=True)
        loss = combined_loss(predictions, batch, 100.0, 300.0, 600.0)
        loss.backward()
        self.assertAlmostEqual(float(predictions.grad[0]), 0.0)
        self.assertLess(float(predictions.grad[1]), 0.0)

        score_only = weighted_huber_loss(predictions.detach(), batch, 100.0)
        self.assertGreater(float(loss.detach()), float(score_only))
        unknown_batch = collate_samples(samples[:1])
        self.assertEqual(
            float(combined_loss(torch.zeros(1), unknown_batch, 100.0, 300.0, 600.0)),
            float(weighted_huber_loss(torch.zeros(1), unknown_batch, 100.0)),
        )
        draw_batch = collate_samples(
            [TrainingSample(encode_fen(INITIAL_FEN), 0.0, result=0)]
        )
        self.assertTrue(bool(draw_batch.result_known[0]))
        self.assertGreater(
            float(combined_loss(torch.tensor([600.0]), draw_batch, 100.0, 300.0, 600.0)),
            float(weighted_huber_loss(torch.tensor([600.0]), draw_batch, 100.0)),
        )

    def test_invalid_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for bad_result in (2, True, "1"):
                path.write_text(
                    json.dumps({"fen": INITIAL_FEN, "score": 0, "result": bad_result}) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "result must be"):
                    JsonlPositionDataset(path)

    def test_legacy_narrow_labels_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"fen": INITIAL_FEN, "score": score})
                    for score in (0, -2, 3)
                ) + "\n",
                encoding="utf-8",
            )
            dataset = JsonlPositionDataset(
                path, quarantine_sources={source_id(path)}, narrow_score_limit=2
            )
        self.assertEqual(dataset.quarantined_count, 2)
        self.assertEqual([sample.target for sample in dataset.samples], [3.0])

    def test_group_split_keeps_games_and_mirrors_apart(self) -> None:
        left = "3k5/9/9/9/9/9/9/2R6/9/5K3 w"
        right = "5k3/9/9/9/9/9/9/6R2/9/3K5 w"
        self.assertEqual(canonical_position_key(left), canonical_position_key(right))
        samples = [
            TrainingSample(
                encode_fen(INITIAL_FEN), 0.0,
                group_key=f"batch-a#game={index}",
                position_key=f"unique-{index}",
            )
            for index in range(50)
        ]
        samples.append(TrainingSample(
            encode_fen(INITIAL_FEN), 0.0,
            group_key="batch-b#game=0", position_key="fixed-test-position",
        ))
        samples.append(TrainingSample(
            encode_fen(INITIAL_FEN), 0.0,
            group_key="batch-c#game=0", position_key="fixed-test-position",
        ))
        manifest = {"groups": ["batch-b#game=0"], "position_keys": ["fixed-test-position"]}
        train, validation, test, dropped = split_indices(samples, 0.2, 2026, manifest)
        self.assertTrue(train and validation and test)
        self.assertEqual(test, [50])
        self.assertEqual(dropped, 1)
        self.assertFalse(
            {samples[i].group_key for i in train}
            & {samples[i].group_key for i in validation}
        )

    def test_arena_does_not_count_unfinished_as_draw(self) -> None:
        games = [
            {"candidate_side": "red", "outcome": "red_win"},
            {"candidate_side": "black", "outcome": "draw"},
            {"candidate_side": "red", "outcome": "unfinished"},
        ]
        summary = summarize_games(games, required_score=0.7, max_unfinished=0, min_games=3)
        self.assertEqual((summary["wins"], summary["draws"], summary["unfinished"]), (1, 1, 1))
        self.assertFalse(summary["passed"])

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
                score_scale=1.0,
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

    def test_scaled_and_legacy_checkpoint_export(self) -> None:
        engine = Path("build/make/xiangqi_cli")
        if not engine.is_file():
            self.skipTest("C++ CLI is not built")
        model = self._make_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.pt"
            torch.save({
                "format": LEGACY_CHECKPOINT_FORMAT,
                "architecture": architecture_metadata(),
                "model_state_dict": model.state_dict(),
            }, legacy)
            self.assertEqual(load_model(legacy).score_scale, 1.0)
            original_score = python_score(model, ASYMMETRIC_FEN)
            with torch.no_grad():
                model.output.weight.div_(600.0)
                model.output.bias.div_(600.0)
            model.score_scale = 600.0
            scaled = root / "scaled.pt"
            nnue = root / "scaled.nnue"
            save_checkpoint(scaled, {"model_state_dict": model.state_dict()}, score_scale=600)
            loaded = load_model(scaled)
            self.assertEqual(loaded.score_scale, 600.0)
            self.assertAlmostEqual(python_score(loaded, ASYMMETRIC_FEN), original_score, delta=1.0e-3)
            export_checkpoint(scaled, nnue)
            self.assertAlmostEqual(cpp_score(engine, nnue, ASYMMETRIC_FEN), original_score, delta=1.0e-3)


class IterationOrchestrationTests(unittest.TestCase):
    def test_state_round_trip_and_replay_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            for generation in range(4):
                generation_dir = work_dir / f"generation-{generation:03d}"
                generation_dir.mkdir()
                (generation_dir / "data.jsonl").write_text(
                    json.dumps({"fen": INITIAL_FEN, "score": generation}) + "\n",
                    encoding="utf-8",
                )

            state_path = work_dir / "state.json"
            atomic_write_json(
                state_path,
                {
                    "format": STATE_FORMAT,
                    "completed_generation": 3,
                    "champion_checkpoint": "/tmp/champion.pt",
                    "champion_model": "/tmp/champion.nnue",
                    "champion_epoch": 40,
                },
            )
            state = load_state(state_path)
            replay = replay_datasets(work_dir, generation=3, count=2)

        self.assertEqual(state.completed_generation, 3)
        self.assertEqual(state.champion_epoch, 40)
        self.assertEqual(
            [path.parent.name for path in replay],
            ["generation-002", "generation-003"],
        )

    def test_invalid_iteration_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text('{"format":"unknown"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported iteration state"):
                load_state(state_path)


if __name__ == "__main__":
    unittest.main()
