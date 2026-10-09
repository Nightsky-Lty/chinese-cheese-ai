"""视觉棋盘到持久化 GameHistory 的集成测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from gui.board_state import BoardState
from gui.observer import GameObserver
from gui.protocol_client import ProtocolEngineClient


INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
AFTER_HORSE = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b"
AFTER_CANNON_SETUP = (
    "rnbakabnr/9/1c5c1/p3p1p1p/2p6/9/P1P1P1P1P/2C4C1/9/RNBAKABNR w"
)
AFTER_FAST_ROOK_REPLY = (
    "rnbakabr1/9/1c5c1/p3p1p1p/2p6/9/P1P1P1P1P/2C6/9/RNBAKABNR w"
)


class ObserverTests(unittest.TestCase):
    def test_observer_accepts_stable_legal_move(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(INITIAL_FEN)
        moved = BoardState.from_fen(AFTER_HORSE)
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=2, search_depth=1
            )
            self.assertIsNone(observer.process_board(initial))
            initialized = observer.process_board(initial)
            self.assertIsNotNone(initialized)
            assert initialized is not None
            self.assertEqual(initialized.kind, "initialized")
            self.assertIsNone(observer.process_board(moved))
            accepted = observer.process_board(moved)
            self.assertIsNotNone(accepted)
            assert accepted is not None
            self.assertEqual(accepted.kind, "move")
            self.assertEqual(accepted.move, "b0c2")
            self.assertEqual(client.state().ply, 1)

    def test_observer_rejects_visual_noise_without_advancing_history(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(INITIAL_FEN)
        noisy_pieces = list(initial.pieces)
        noisy_pieces[40] = "P"
        noisy_pieces[41] = "P"
        noisy = BoardState(tuple(noisy_pieces))
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=1, search_depth=1
            )
            observer.process_board(initial)
            rejected = observer.process_board(noisy)
            self.assertIsNotNone(rejected)
            assert rejected is not None
            self.assertEqual(rejected.kind, "rejected")
            self.assertEqual(client.state().ply, 0)

    def test_observer_ignores_one_square_intermediate_then_accepts_move(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(INITIAL_FEN)
        intermediate_pieces = list(initial.pieces)
        intermediate_pieces[1] = "."
        intermediate = BoardState(tuple(intermediate_pieces))
        moved = BoardState.from_fen(AFTER_HORSE)
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=1, search_depth=1
            )
            observer.process_board(initial)

            transient = observer.process_board(intermediate)
            self.assertIsNotNone(transient)
            assert transient is not None
            self.assertEqual(transient.kind, "transient")
            self.assertEqual(transient.changes, ("b0 N->.",))
            self.assertEqual(observer.board, initial)
            self.assertEqual(client.state().ply, 0)

            accepted = observer.process_board(moved)
            self.assertIsNotNone(accepted)
            assert accepted is not None
            self.assertEqual(accepted.kind, "move")
            self.assertEqual(accepted.move, "b0c2")
            self.assertEqual(client.state().ply, 1)

    def test_partial_board_recovers_fast_overlapping_capture_reply(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(AFTER_CANNON_SETUP)
        final = BoardState.from_fen(AFTER_FAST_ROOK_REPLY)
        partial_pieces = list(final.pieces)
        partial_pieces[9 * 9 + 7] = "?"  # h9 落子光效遮挡黑车。
        partial = BoardState(tuple(partial_pieces))
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=1, search_depth=1
            )
            observer.process_board(initial)

            recovered = observer.process_board(partial, expected_move="h2h9")

            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.kind, "moves")
            self.assertEqual(recovered.moves, ("h2h9", "i9h9"))
            self.assertEqual(observer.board, final)
            self.assertEqual(observer.side_to_move, "red")
            self.assertEqual(client.state().ply, 2)


if __name__ == "__main__":
    unittest.main()
