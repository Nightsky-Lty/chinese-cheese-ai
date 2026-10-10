"""用假文本协议引擎验证 arena 的终局、限时、失败持久化与门禁。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.arena import (
    Opponent,
    ProtocolEngine,
    parse_args as parse_arena_args,
    resolve_search_depth,
    run_arena,
    summarize_games,
)
from training.baseline import parse_args as parse_baseline_args, validation_gate
from training.checkpoint import save_checkpoint
from training.iterate import (
    IterationState,
    STATE_FORMAT,
    arena_opponents,
    candidate_validation_evidence,
    parse_args as parse_iterate_args,
    promote_candidate,
)
from training.model import HalfKANetwork


FAKE_PROTOCOL = r'''#!/usr/bin/env python3
import os
import sys
import time

terminal_at = int(os.environ.get("FAKE_TERMINAL_AT", "3"))
mode = os.environ.get("FAKE_PROTOCOL_MODE", "normal")
log_path = os.environ.get("FAKE_PROTOCOL_LOG")
ply = 0
print("ready xiangqi-protocol 1", flush=True)
for line in sys.stdin:
    line = line.rstrip("\n")
    if line == "quit":
        print("bye", flush=True)
        break
    if line.startswith("position fen "):
        if mode == "fail_position":
            sys.exit(7)
        ply = 0
        print("ok position", flush=True)
    elif line.startswith("move "):
        ply += 1
        print("ok move", flush=True)
    elif line == "result":
        if terminal_at >= 0 and ply >= terminal_at:
            print("result red_win reason checkmate", flush=True)
        else:
            print("result ongoing reason none", flush=True)
    elif line.startswith("go "):
        if log_path:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        if mode == "sleep_go":
            time.sleep(2)
        elif mode == "partial_go":
            sys.stdout.write("bestmove")
            sys.stdout.flush()
            time.sleep(2)
        else:
            print("bestmove a3a4 score 0 depth 2 nodes 1 timedout 0", flush=True)
    else:
        print("error unsupported", flush=True)
'''


class ArenaPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = self.root / "fake_protocol.py"
        self.engine.write_text(FAKE_PROTOCOL, encoding="utf-8")
        self.engine.chmod(0o755)
        self.candidate = self.root / "candidate.nnue"
        self.candidate.write_bytes(b"fake candidate network")
        self.openings = self.root / "openings.json"
        self.openings.write_text(
            json.dumps({
                "format": "test",
                "initial_fen": "fake w",
                "openings": [{"name": "two-ply", "moves": ["b2e2", "b7e7"]}],
            }),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_terminal_after_last_permitted_ply_is_decided(self) -> None:
        log = self.root / "go.log"
        output = self.root / "arena.json"
        with patch.dict(os.environ, {"FAKE_PROTOCOL_LOG": str(log), "FAKE_TERMINAL_AT": "3"}):
            report = run_arena(
                self.engine, self.candidate, [Opponent("hand", None)], output,
                depth=2, time_limit_ms=17, max_plies=3,
                required_score=0.0, min_games=2, min_independent_pairs=1,
                openings_path=self.openings, timeout_seconds=1,
            )
        result = report["opponents"]["hand"]
        self.assertEqual(result["summary"]["games"], 2)
        self.assertEqual(result["summary"]["unfinished"], 0)
        self.assertTrue(all(game["plies"] == 3 for game in result["games"]))
        self.assertTrue(all(game["reason"] == "checkmate" for game in result["games"]))
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [
            "go depth 2 movetime 17", "go depth 2 movetime 17",
        ])
        # One paired opening produces a full-width conservative interval and
        # cannot pass the formal 100-game / 50-opening floor.
        self.assertEqual(result["summary"]["score_fraction_ci95"]["lower"], 0.0)
        self.assertFalse(report["passed"])

    def test_hung_engine_and_partial_stdout_are_killed_by_deadline(self) -> None:
        for mode in ("sleep_go", "partial_go"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"FAKE_PROTOCOL_MODE": mode}):
                engine = ProtocolEngine(self.engine, self.candidate, timeout_seconds=0.05)
                with self.assertRaises(TimeoutError):
                    engine.ask("go depth 2 movetime 10")
                self.assertIsNotNone(engine.process.poll())
                engine.close()

    def test_depth_mode_applies_same_depth_without_a_time_limit(self) -> None:
        log = self.root / "depth-go.log"
        with patch.dict(os.environ, {"FAKE_PROTOCOL_LOG": str(log)}):
            report = run_arena(
                self.engine, self.candidate, [Opponent("hand", None)],
                self.root / "depth-arena.json",
                search_mode="depth", depth=2, time_limit_ms=None, max_plies=3,
                required_score=0.0, min_games=2, min_independent_pairs=1,
                openings_path=self.openings, timeout_seconds=1,
            )
        self.assertEqual(report["config"]["search_mode"], "depth")
        self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [
            "go depth 2", "go depth 2",
        ])

    def test_cli_time_defaults_use_high_depth_cap_and_preserve_explicit_caps(self) -> None:
        with patch("sys.argv", ["arena", "--candidate", "candidate.nnue", "--output", "arena.json"]):
            arena_args = parse_arena_args()
        with patch("sys.argv", ["iterate"]):
            iterate_args = parse_iterate_args()
        with patch("sys.argv", ["baseline", "teacher.jsonl", "--work-dir", "run"]):
            baseline_args = parse_baseline_args()

        self.assertEqual(resolve_search_depth(arena_args.search_mode, arena_args.depth), 64)
        self.assertEqual(resolve_search_depth(iterate_args.arena_mode, iterate_args.arena_depth), 64)
        self.assertEqual(resolve_search_depth(baseline_args.search_mode, baseline_args.depth), 64)
        self.assertEqual(resolve_search_depth("depth", None), 4)
        self.assertEqual(resolve_search_depth("time", 2), 2)

    def test_protocol_failure_is_recorded_and_report_cannot_be_overwritten(self) -> None:
        output = self.root / "failed-arena.json"
        with patch.dict(os.environ, {"FAKE_PROTOCOL_MODE": "fail_position"}):
            report = run_arena(
                self.engine, self.candidate, [Opponent("hand", None)], output,
                depth=1, time_limit_ms=10, max_plies=4,
                required_score=0.0, min_games=2, min_independent_pairs=1,
                openings_path=self.openings, timeout_seconds=0.2,
            )
        self.assertTrue(output.is_file())
        self.assertFalse(report["passed"])
        games = report["opponents"]["hand"]["games"]
        self.assertEqual(games[0]["reason"], "protocol_failure")
        self.assertTrue(games[0]["failure"])
        with self.assertRaises(FileExistsError):
            run_arena(
                self.engine, self.candidate, [Opponent("hand", None)], output,
                min_games=2, min_independent_pairs=1,
                openings_path=self.openings,
            )

    def test_minimum_sample_and_paired_interval_gate(self) -> None:
        games = []
        for index in range(50):
            for side in ("red", "black"):
                games.append({
                    "opening": f"opening-{index}",
                    "pair_id": index,
                    "candidate_side": side,
                    "outcome": f"{side}_win",
                })
        passed = summarize_games(games, 0.70, 0, 100, 50)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["score_fraction_ci95"]["independent_openings"], 50)
        self.assertLess(passed["score_fraction_ci95"]["lower"], 1.0)
        self.assertGreaterEqual(passed["score_fraction_ci95"]["lower"], 0.70)
        small = summarize_games(games[:12], 0.0, 0, 12, 50)
        self.assertFalse(small["passed"])
        self.assertEqual(small["complete_paired_openings"], 6)

    def test_unfinished_side_is_excluded_from_paired_confidence_clusters(self) -> None:
        games = [
            {"opening": "incomplete", "pair_id": 0, "candidate_side": "red", "outcome": "red_win"},
            {"opening": "incomplete", "pair_id": 0, "candidate_side": "black", "outcome": "unfinished"},
            {"opening": "complete", "pair_id": 1, "candidate_side": "red", "outcome": "red_win"},
            {"opening": "complete", "pair_id": 1, "candidate_side": "black", "outcome": "black_win"},
        ]
        summary = summarize_games(games, 0.0, 1, 4, 1)
        self.assertEqual(summary["complete_paired_openings"], 1)
        self.assertEqual(summary["score_fraction_ci95"]["independent_openings"], 1)
        self.assertFalse(summary["passed"])

    def test_hand_is_always_an_iteration_opponent(self) -> None:
        self.assertEqual([item.name for item in arena_opponents(None, [])], ["hand"])
        history = self.root / "history.nnue"
        history.touch()
        self.assertEqual(
            [item.name for item in arena_opponents(history, [history])],
            ["hand", "champion", "history-1"],
        )

    def test_score_training_best_checkpoint_schema_and_fixed_test_gate(self) -> None:
        checkpoint_path = self.root / "best.pt"
        split = {
            "version": 1,
            "split_seed": 2026,
            "counts": {"train": 70, "validation": 20, "test": 10},
        }
        # This is the real best.pt payload shape written by training.train:
        # best_validation_loss + validation_metrics + epoch (no validation_loss).
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": HalfKANetwork().state_dict(),
                "epoch": 7,
                "best_epoch": 7,
                "best_validation_loss": 0.12,
                "validation_metrics": {"total_loss": 0.12},
                "split_metadata": split,
            },
            score_scale=600.0,
        )
        from training.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        self.assertNotIn("validation_loss", checkpoint)
        report = {
            "format": "xiangqi-nnue-training-report-v1",
            "config": {"split": split},
            "best_epoch": 7,
            "best_validation_loss": 0.12,
            "fixed_test_evaluations": 1,
            "fixed_test": {"sample_count": 10, "score_loss": 0.13},
        }
        evidence = candidate_validation_evidence(report, checkpoint)
        self.assertTrue(evidence["passed"], evidence)
        self.assertEqual(evidence["checkpoint_validation_loss"], 0.12)

        report["config"]["split"]["split_seed"] = 9
        self.assertFalse(candidate_validation_evidence(report, checkpoint)["passed"])
        report["config"]["split"]["split_seed"] = 2026
        report["best_epoch"] = 8
        self.assertFalse(candidate_validation_evidence(report, checkpoint)["passed"])
        report["best_epoch"] = 7
        report["best_validation_loss"] = 0.121
        self.assertFalse(candidate_validation_evidence(report, checkpoint)["passed"])
        report["best_validation_loss"] = 0.12

        report["fixed_test_evaluations"] = 0
        self.assertFalse(candidate_validation_evidence(report, checkpoint)["passed"])
        report["fixed_test_evaluations"] = 1
        report["fixed_test"]["sample_count"] = 0
        self.assertFalse(candidate_validation_evidence(report, checkpoint)["passed"])

    def test_champion_changes_only_after_all_three_gates_pass(self) -> None:
        state_path = self.root / "state.json"
        old_checkpoint = self.root / "old.pt"
        old_model = self.root / "old.nnue"
        old_checkpoint.write_bytes(b"existing champion checkpoint")
        old_model.write_bytes(b"existing champion model")
        (self.root / "champion.pt").write_bytes(b"existing champion checkpoint")
        (self.root / "champion.nnue").write_bytes(b"existing champion model")
        state_path.write_text(
            json.dumps({
                "format": STATE_FORMAT,
                "completed_generation": 2,
                "champion_checkpoint": str(old_checkpoint),
                "champion_model": str(old_model),
                "champion_epoch": 4,
            }),
            encoding="utf-8",
        )
        old_state_bytes = state_path.read_bytes()
        candidate_checkpoint = self.root / "candidate.pt"
        save_checkpoint(
            candidate_checkpoint,
            {"model_state_dict": HalfKANetwork().state_dict(), "epoch": 5},
            score_scale=600.0,
        )
        candidate_model = self.root / "candidate.nnue"
        candidate_model.write_bytes(b"candidate model")
        good_validation = {"passed": True}
        passing_arena = {"passed": True}
        failed_gate_sets = [
            ({"passed": False}, True, passing_arena),
            (good_validation, False, passing_arena),
            (good_validation, True, {"passed": False}),
        ]
        for validation, parity, arena in failed_gate_sets:
            with self.subTest(validation=validation, parity=parity, arena=arena):
                self.assertIsNone(promote_candidate(
                    state_path, 3, candidate_checkpoint, candidate_model,
                    validation, parity, arena,
                ))
                self.assertEqual(state_path.read_bytes(), old_state_bytes)
                self.assertEqual((self.root / "champion.pt").read_bytes(), b"existing champion checkpoint")
                self.assertEqual((self.root / "champion.nnue").read_bytes(), b"existing champion model")

        promoted = promote_candidate(
            state_path, 3, candidate_checkpoint, candidate_model,
            good_validation, True, passing_arena,
        )
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.completed_generation, 3)
        self.assertEqual(promoted.champion_epoch, 5)
        self.assertEqual((self.root / "champion.pt").read_bytes(), candidate_checkpoint.read_bytes())
        self.assertEqual((self.root / "champion.nnue").read_bytes(), candidate_model.read_bytes())

    def test_baseline_gate_requires_fixed_test_sample_count_to_match_manifest(self) -> None:
        checkpoint = self.root / "baseline-best.pt"
        split = {
            "version": 1,
            "split_seed": 2026,
            "counts": {"train": 80, "validation": 10, "test": 8},
        }
        save_checkpoint(
            checkpoint,
            {
                "model_state_dict": HalfKANetwork().state_dict(),
                "epoch": 1,
                "best_epoch": 1,
                "best_validation_loss": 0.2,
                "validation_metrics": {"total_loss": 0.2},
                "split_metadata": split,
            },
            score_scale=600.0,
        )
        report_path = self.root / "training-report.json"
        report = {
            "format": "xiangqi-nnue-training-report-v1",
            "config": {"split": split},
            "best_epoch": 1,
            "best_validation_loss": 0.2,
            "fixed_test_evaluations": 1,
            "fixed_test": {"sample_count": 7, "score_loss": 0.3},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.assertFalse(validation_gate(report_path, checkpoint)["passed"])
        report["fixed_test"]["sample_count"] = 8
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.assertTrue(validation_gate(report_path, checkpoint)["passed"])
        report["best_epoch"] = 0
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.assertFalse(validation_gate(report_path, checkpoint)["passed"])


if __name__ == "__main__":
    unittest.main()
