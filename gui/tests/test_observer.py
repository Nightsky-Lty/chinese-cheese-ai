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

    def test_single_clean_frame_recovers_amid_incompatible_animation_frames(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(INITIAL_FEN)
        artifact_pieces = list(initial.pieces)
        artifact_pieces[2 * 9 + 1] = "."  # b2
        artifact_pieces[2 * 9 + 2] = "C"  # c2
        artifact_pieces[5 * 9 + 2] = "r"  # c5 动画假棋子
        artifact_pieces[6 * 9 + 1] = "?"  # b6 光效
        artifact = BoardState(tuple(artifact_pieces))

        final_pieces = list(initial.pieces)
        final_pieces[2 * 9 + 1] = "."
        final_pieces[2 * 9 + 2] = "C"
        final_pieces[9 * 9 + 6] = "?"  # g9 源格仍被光效覆盖
        final_pieces[7 * 9 + 4] = "b"  # e7 已识别到黑象
        final = BoardState(tuple(final_pieces))

        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=4, search_depth=1
            )
            for _ in range(3):
                self.assertIsNone(observer.process_board(initial))
            self.assertEqual(observer.process_board(initial).kind, "initialized")

            # 动画假帧无法形成合法序列，也没有累计到四帧。
            self.assertIsNone(
                observer.process_board(artifact, expected_move="b2c2")
            )

            # 无需等待四张完全一致的最终截图，唯一合法序列可在单帧确认。
            recovered = observer.process_board(final, expected_move="b2c2")
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.kind, "moves")
            self.assertEqual(recovered.moves, ("b2c2", "g9e7"))
            self.assertEqual(client.state().ply, 2)

    def test_single_partial_frame_recovers_opponent_move_after_confirmation(self) -> None:
        """提示圈遮住无关空格时，已知起终点仍能确认对手着法。"""

        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        with ProtocolEngineClient(executable) as client:
            client.set_position(INITIAL_FEN)
            for move in ("b2c2", "c9e7", "c2c6", "h9i7", "h2i2"):
                client.play(move)
            before_reply = BoardState.from_fen(client.state().fen)
            final = BoardState.from_fen(client.play("h7f7").fen)
            client.undo()

            observer = GameObserver(
                client, initial_side="black", stable_frames=4, search_depth=1
            )
            observer.board = before_reply
            observer.stability.restore_accepted(before_reply)

            partial_pieces = list(final.pieces)
            partial_pieces[9 * 9 + 2] = "?"  # c9 提示圈覆盖空格。
            partial_pieces[9 * 9 + 7] = "?"  # h9 提示圈覆盖空格。
            partial = BoardState(tuple(partial_pieces))

            recovered = observer.process_board(partial)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.kind, "move")
            self.assertEqual(recovered.move, "h7f7")
            self.assertEqual(observer.board, final)
            self.assertEqual(observer.side_to_move, "red")
            self.assertEqual(client.state().ply, 6)

    def test_partial_opponent_frame_needs_both_changed_squares(self) -> None:
        """落点未知时仅凭起点消失，不应猜测对手走法。"""

        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        with ProtocolEngineClient(executable) as client:
            client.set_position(INITIAL_FEN)
            before = BoardState.from_fen(client.state().fen)
            final = BoardState.from_fen(client.play("b0c2").fen)
            client.undo()
            observer = GameObserver(
                client, initial_side="red", stable_frames=4, search_depth=1
            )
            observer.board = before
            observer.stability.restore_accepted(before)
            pieces = list(final.pieces)
            pieces[2 * 9 + 2] = "?"  # c2 落点尚未识别。

            self.assertIsNone(observer.process_board(BoardState(tuple(pieces))))
            self.assertEqual(client.state().ply, 0)
            self.assertEqual(observer.board, before)

    def test_extra_animation_pieces_wait_instead_of_rejecting_expected_move(self) -> None:
        """预期马步已出现、画面却多出两个兵时继续等待干净截图。"""

        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        fen = "4k4/9/9/4p4/9/9/4N4/9/9/4K4 w"
        initial = BoardState.from_fen(fen)
        after_move = GameObserver._apply_move(initial, "e3f5")
        pieces = list(after_move.pieces)
        pieces[4 * 9 + 4] = "P"  # e4 动画误报。
        pieces[5 * 9 + 4] = "P"  # e5 动画误报。
        noisy = BoardState(tuple(pieces))
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=2, search_depth=1
            )
            observer.process_board(initial)
            self.assertEqual(observer.process_board(initial).kind, "initialized")
            self.assertIsNone(observer.process_board(noisy, expected_move="e3f5"))
            transient = observer.process_board(noisy, expected_move="e3f5")
            self.assertIsNotNone(transient)
            assert transient is not None
            self.assertEqual(transient.kind, "transient")
            self.assertEqual(client.state().ply, 0)

            self.assertIsNone(observer.process_board(after_move, expected_move="e3f5"))
            confirmed = observer.process_board(after_move, expected_move="e3f5")
            self.assertIsNotNone(confirmed)
            assert confirmed is not None
            self.assertEqual(confirmed.kind, "move")
            self.assertEqual(confirmed.move, "e3f5")
            self.assertEqual(client.state().ply, 1)


if __name__ == "__main__":
    unittest.main()
