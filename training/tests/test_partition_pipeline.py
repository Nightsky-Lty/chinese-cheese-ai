"""固定测试集、对局分区和来源隔离的回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset import JsonlPositionDataset, TrainingSample, collate_samples
from training.halfka import encode_fen
from training.partition import (
    DEFAULT_TEST_MANIFEST,
    canonical_position_key,
    create_test_manifest,
    game_group_key,
    load_test_manifest,
    source_id,
    split_indices,
    stable_fraction,
    write_test_manifest,
)

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
LEFT_FEN = "3k5/9/9/9/9/9/9/2R6/9/5K3 w"
RIGHT_FEN = "5k3/9/9/9/9/9/9/6R2/9/3K5 w"


def make_sample(group: str, position: str, fen: str = INITIAL_FEN) -> TrainingSample:
    return TrainingSample(
        encode_fen(fen), 0.0, fen=fen, group_key=group, position_key=position
    )


def region_groups(samples: list[TrainingSample], *regions: list[int]) -> list[set[str]]:
    return [
        {samples[index].group_key for index in indices}
        for indices in regions
    ]


class PartitionPipelineTests(unittest.TestCase):
    def test_games_stay_in_one_region_and_mirrors_do_not_cross_regions(self) -> None:
        seed = 2026
        sample_groups = [f"batch-a#game={index}" for index in range(60)]
        validation_group = next(
            group for group in sample_groups
            if stable_fraction(seed + 1, group) < 0.2
        )
        training_group = next(
            group for group in sample_groups
            if stable_fraction(seed + 1, group) >= 0.2
        )
        samples = [
            make_sample(group, f"position-{group}-{ply}")
            for group in sample_groups
            for ply in range(2)
        ]
        mirror_key = canonical_position_key(LEFT_FEN)
        samples.extend(
            [
                make_sample(validation_group, mirror_key, LEFT_FEN),
                make_sample(training_group, mirror_key, RIGHT_FEN),
            ]
        )

        train, validation, test, dropped = split_indices(samples, 0.2, seed, None)
        self.assertTrue(train and validation)
        self.assertFalse(test)
        groups_by_region = region_groups(samples, train, validation, test)
        self.assertFalse(groups_by_region[0] & groups_by_region[1])
        self.assertFalse(groups_by_region[0] & groups_by_region[2])
        self.assertFalse(groups_by_region[1] & groups_by_region[2])
        positions_by_region = [
            {samples[index].position_key for index in indices}
            for indices in (train, validation, test)
        ]
        self.assertFalse(positions_by_region[0] & positions_by_region[1])
        self.assertEqual(dropped, 1)

    def test_fixed_test_replay_position_is_removed_without_dropping_its_game(self) -> None:
        samples = [
            make_sample("teacher#game=4", "fixed-test-position"),
            make_sample("future-replay#game=9", "fixed-test-position"),
            make_sample("future-replay#game=9", "replay-only-position"),
        ]
        manifest = {
            "groups": ["teacher#game=4"],
            "position_keys": ["fixed-test-position"],
        }

        train, validation, test, dropped = split_indices(samples, 0.0, 2026, manifest)
        self.assertEqual(test, [0])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(train) + len(validation), 1)
        self.assertIn(samples[train[0] if train else validation[0]].position_key,
                      {"replay-only-position"})

    def test_batch_paths_prevent_game_id_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "batch-a.jsonl"
            second = root / "batch-b.jsonl"
            for path in (first, second):
                path.write_text(
                    json.dumps({"fen": INITIAL_FEN, "score": 0, "game": 0}) + "\n",
                    encoding="utf-8",
                )
            first_group = game_group_key(first, {"game": 0})
            second_group = game_group_key(second, {"game": 0})

        self.assertNotEqual(first_group, second_group)

    def test_source_change_invalidates_bound_fixed_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "teacher.jsonl"
            manifest_path = root / "fixed-test.json"
            records = [
                {"fen": INITIAL_FEN, "score": 10, "game": 0},
                {"fen": LEFT_FEN, "score": 20, "game": 1},
            ]
            data_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            dataset = JsonlPositionDataset(data_path)
            payload = create_test_manifest(dataset.samples, 0.99, 2026)
            write_test_manifest(manifest_path, payload)
            loaded = load_test_manifest(manifest_path)
            self.assertEqual(loaded["groups"], payload["groups"])
            self.assertEqual(loaded["source_fingerprints"][source_id(data_path)],
                             dataset.source_fingerprint)

            data_path.write_text(
                json.dumps({"fen": INITIAL_FEN, "score": 11, "game": 0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_test_manifest(manifest_path)

    def test_production_test_manifest_keeps_existing_identity_and_sources(self) -> None:
        manifest = load_test_manifest(DEFAULT_TEST_MANIFEST)
        self.assertEqual(manifest["format"], "xiangqi-fixed-test-v1")
        self.assertEqual(len(manifest["groups"]), 244)
        self.assertEqual(len(manifest["position_keys"]), 10_994)
        self.assertEqual(len(manifest["source_fingerprints"]), 13)

    def test_unknown_results_and_normal_near_zero_scores_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path = root / "legacy.jsonl"
            normal_path = root / "normal.jsonl"
            legacy_path.write_text(
                "\n".join(
                    json.dumps({"fen": INITIAL_FEN, "score": score, "game": 0})
                    for score in (0, 2, 3)
                ) + "\n",
                encoding="utf-8",
            )
            normal_path.write_text(
                json.dumps({"fen": INITIAL_FEN, "score": 0, "result": None, "game": 0})
                + "\n",
                encoding="utf-8",
            )
            legacy = JsonlPositionDataset(
                legacy_path,
                quarantine_sources={source_id(legacy_path)},
                narrow_score_limit=2,
            )
            normal = JsonlPositionDataset(normal_path)
            batch = collate_samples(normal.samples)

        self.assertEqual(legacy.quarantined_count, 2)
        self.assertEqual([sample.target for sample in legacy.samples], [3.0])
        self.assertEqual([sample.target for sample in normal.samples], [0.0])
        self.assertIsNone(normal.samples[0].result)
        self.assertEqual(batch.result_known.tolist(), [False])
        self.assertEqual(batch.phases, ["opening"])

    def test_mixed_missing_game_ids_cannot_split_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"fen": INITIAL_FEN, "score": 0, "game": 0}),
                        json.dumps({"fen": LEFT_FEN, "score": 0}),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            dataset = JsonlPositionDataset(path)

        self.assertEqual(len({sample.group_key for sample in dataset.samples}), 1)
        self.assertTrue(dataset.samples[0].group_key.endswith("#file"))


if __name__ == "__main__":
    unittest.main()
