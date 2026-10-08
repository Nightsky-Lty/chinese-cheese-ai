"""连续对局状态机及落子确认测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

from gui.board_state import BoardState
from gui.game_controller import ControllerPhase, GameController
from gui.observer import GameObserver, ObserverEvent
from gui.protocol_client import ProtocolEngineClient


INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def apply_coordinate_move(board: BoardState, move: str) -> BoardState:
    """在测试棋盘上应用一个不检查规则的四字符走法。"""

    source = int(move[1]) * 9 + ord(move[0]) - ord("a")
    target = int(move[3]) * 9 + ord(move[2]) - ord("a")
    pieces = list(board.pieces)
    pieces[target] = pieces[source]
    pieces[source] = "."
    return BoardState(tuple(pieces))


class FakeObserver:
    """按测试预设顺序返回观察器事件。"""

    def __init__(self, side_to_move: str, events: list[tuple[ObserverEvent, str]]) -> None:
        self.side_to_move = side_to_move
        self.events = events

    def process_board(self, _board: BoardState) -> ObserverEvent | None:
        if not self.events:
            return None
        event, next_side = self.events.pop(0)
        self.side_to_move = next_side
        return event


class GameControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = cast(BoardState, object())

    def test_requests_one_move_then_waits_for_visual_confirmation(self) -> None:
        observer = FakeObserver(
            "red",
            [
                (ObserverEvent("initialized", best_move="b0c2"), "red"),
                (ObserverEvent("move", move="b0c2", best_move="b9c7"), "black"),
            ],
        )
        requested: list[str] = []
        controller = GameController(
            cast(GameObserver, observer), ai_side="red", move_executor=requested.append
        )

        first = controller.process_board(self.board)
        self.assertEqual(requested, ["b0c2"])
        self.assertEqual(first[-1].kind, "move_requested")
        self.assertEqual(controller.phase, ControllerPhase.AWAITING_CONFIRMATION)

        second = controller.process_board(self.board)
        self.assertEqual(second[0].kind, "move_confirmed")
        self.assertEqual(controller.phase, ControllerPhase.WAITING_OPPONENT)
        self.assertIsNone(controller.pending_move)
        self.assertEqual(requested, ["b0c2"])

    def test_opponent_move_triggers_ai_reply(self) -> None:
        observer = FakeObserver(
            "black",
            [
                (ObserverEvent("initialized", best_move="b9c7"), "black"),
                (ObserverEvent("move", move="b9c7", best_move="b0c2"), "red"),
            ],
        )
        requested: list[str] = []
        controller = GameController(
            cast(GameObserver, observer), ai_side="red", move_executor=requested.append
        )

        controller.process_board(self.board)
        self.assertEqual(controller.phase, ControllerPhase.WAITING_OPPONENT)
        events = controller.process_board(self.board)
        self.assertEqual(
            [event.kind for event in events], ["opponent_move", "move_requested"]
        )
        self.assertEqual(requested, ["b0c2"])

    def test_mismatched_executed_move_pauses_without_retry(self) -> None:
        observer = FakeObserver(
            "red",
            [
                (ObserverEvent("initialized", best_move="b0c2"), "red"),
                (ObserverEvent("move", move="h0g2", best_move="b9c7"), "black"),
            ],
        )
        requested: list[str] = []
        controller = GameController(
            cast(GameObserver, observer), ai_side="red", move_executor=requested.append
        )

        controller.process_board(self.board)
        events = controller.process_board(self.board)
        self.assertEqual(events[-1].kind, "paused")
        self.assertIn("b0c2", events[-1].message or "")
        self.assertIn("h0g2", events[-1].message or "")
        self.assertEqual(controller.phase, ControllerPhase.PAUSED)
        controller.process_board(self.board)
        self.assertEqual(requested, ["b0c2"])

    def test_recognition_rejection_and_executor_failure_pause(self) -> None:
        rejected = FakeObserver(
            "red", [(ObserverEvent("rejected", message="visual noise"), "red")]
        )
        controller = GameController(
            cast(GameObserver, rejected),
            ai_side="red",
            move_executor=lambda _move: None,
        )
        self.assertEqual(controller.process_board(self.board)[-1].kind, "paused")

        failing = FakeObserver(
            "red", [(ObserverEvent("initialized", best_move="b0c2"), "red")]
        )

        def fail(_move: str) -> None:
            raise RuntimeError("offline actuator unavailable")

        controller = GameController(
            cast(GameObserver, failing), ai_side="red", move_executor=fail
        )
        events = controller.process_board(self.board)
        self.assertEqual(events[-1].kind, "paused")
        self.assertIn("offline actuator unavailable", events[-1].message or "")

    def test_no_best_move_finishes_game(self) -> None:
        observer = FakeObserver(
            "red",
            [
                (
                    ObserverEvent(
                        "initialized",
                        best_move=None,
                        outcome="draw",
                        reason="threefold_repetition",
                    ),
                    "red",
                )
            ],
        )
        controller = GameController(
            cast(GameObserver, observer),
            ai_side="red",
            move_executor=lambda _move: None,
        )
        events = controller.process_board(self.board)
        self.assertEqual(events[-1].kind, "finished")
        self.assertEqual(events[-1].message, "draw: threefold_repetition")
        self.assertEqual(controller.phase, ControllerPhase.FINISHED)

    def test_unconfirmed_move_pauses_after_frame_limit(self) -> None:
        observer = FakeObserver(
            "red", [(ObserverEvent("initialized", best_move="b0c2"), "red")]
        )
        controller = GameController(
            cast(GameObserver, observer),
            ai_side="red",
            move_executor=lambda _move: None,
            confirmation_frame_limit=2,
        )
        controller.process_board(self.board)
        self.assertEqual(controller.process_board(self.board), ())
        events = controller.process_board(self.board)
        self.assertEqual(events[-1].kind, "paused")
        self.assertIn("2 frames", events[-1].message or "")

    def test_real_observer_search_and_visual_confirmation_loop(self) -> None:
        executable = Path("build/make/xiangqi_protocol")
        if not executable.is_file():
            self.skipTest("xiangqi_protocol has not been built")
        initial = BoardState.from_fen(INITIAL_FEN)
        requested: list[str] = []
        with ProtocolEngineClient(executable) as client:
            observer = GameObserver(
                client, initial_side="red", stable_frames=1, search_depth=1
            )
            controller = GameController(
                observer, ai_side="red", move_executor=requested.append
            )
            events = controller.process_board(initial)
            self.assertEqual(events[-1].kind, "move_requested")
            self.assertEqual(len(requested), 1)

            moved = apply_coordinate_move(initial, requested[0])
            events = controller.process_board(moved)
            self.assertEqual(events[0].kind, "move_confirmed")
            self.assertEqual(controller.phase, ControllerPhase.WAITING_OPPONENT)
            self.assertEqual(client.state().ply, 1)


if __name__ == "__main__":
    unittest.main()
