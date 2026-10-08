"""C++ 引擎输出解析的轻量测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from gui.engine_client import _SEARCH_PATTERN
from gui.protocol_client import ProtocolEngineClient, ProtocolError


class EngineClientTests(unittest.TestCase):
    def test_parse_final_search_line(self) -> None:
        line = (
            "search: alpha-beta/pvs depth 4 score 23 nodes 100 qnodes 50 "
            "cutoffs 12 tthits 3 ttcutoffs 1 ttmoves 2 pvsresearches 0 bestmove b0c2"
        )
        match = _SEARCH_PATTERN.match(line)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group("move"), "b0c2")
        self.assertEqual(int(match.group("score")), 23)

    def test_persistent_protocol_preserves_history(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        with ProtocolEngineClient(executable) as client:
            client.ping()
            initial = client.state()
            self.assertEqual(initial.ply, 0)
            game_result = client.result()
            self.assertEqual(game_result.outcome, "ongoing")
            self.assertFalse(game_result.finished)
            self.assertIn("b0c2", client.legal_moves())

            moved = client.play("b0c2")
            self.assertEqual(moved.ply, 1)
            self.assertEqual(
                moved.fen,
                "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b",
            )
            analysis = client.search(depth=1)
            self.assertIsNotNone(analysis.best_move)
            self.assertEqual(analysis.depth, 1)

            restored = client.undo()
            self.assertEqual(restored.fen, initial.fen)
            self.assertEqual(restored.ply, 0)

    def test_protocol_rejects_illegal_move(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        with ProtocolEngineClient(executable) as client:
            with self.assertRaises(ProtocolError):
                client.play("a0a9")

    def test_protocol_reports_explicit_checkmate_result(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        with ProtocolEngineClient(executable) as client:
            client.set_position("4k4/3PRP3/4P4/9/9/9/9/9/9/4K4 b")
            result = client.result()
            self.assertTrue(result.finished)
            self.assertEqual(result.outcome, "red_win")
            self.assertEqual(result.reason, "checkmate")

    def test_protocol_exposes_history_adjudication_reason(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        cycle = (
            "d8e8",
            "e9f9",
            "e8f8",
            "f9e9",
            "f8e8",
            "e9d9",
            "e8d8",
            "d9e9",
        )
        with ProtocolEngineClient(executable) as client:
            client.set_position("4k4/3R5/9/9/4p4/9/9/9/9/4K4 w")
            for _repetition in range(2):
                for move in cycle:
                    client.play(move)
            result = client.result()
            self.assertEqual(result.outcome, "black_win")
            self.assertEqual(result.reason, "red_perpetual_check")


if __name__ == "__main__":
    unittest.main()
