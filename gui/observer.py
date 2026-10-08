"""将稳定视觉局面、合法走法差分与常驻 C++ 历史串联。"""

from __future__ import annotations

from dataclasses import dataclass

from .board_state import BoardState, StableBoardDetector, detect_move
from .protocol_client import ProtocolEngineClient, ProtocolSearchResult
from .recognition import RecognitionResult


@dataclass(frozen=True)
class ObserverEvent:
    """只读对局观察器在接受稳定局面后产生的事件。"""

    kind: str
    fen: str | None = None
    move: str | None = None
    best_move: str | None = None
    score: int | None = None
    depth: int | None = None
    timed_out: bool = False
    outcome: str | None = None
    reason: str | None = None
    message: str | None = None


class GameObserver:
    """维护可信视觉棋盘，并把真实走法追加到 C++ ``GameHistory``。"""

    def __init__(
        self,
        engine: ProtocolEngineClient,
        *,
        initial_side: str,
        stable_frames: int = 3,
        search_depth: int = 4,
        move_time_ms: int | None = None,
    ) -> None:
        """创建只读观察器。

        参数:
            engine: 已启动的常驻 C++ 协议客户端。
            initial_side: 第一张稳定局面的当前行棋方。
            stable_frames: 接受局面前要求的连续相同帧数。
            search_depth: 每次接受局面后用于提示走法的搜索深度。
            move_time_ms: 每次搜索的可选时间上限，单位毫秒。
        """

        side = initial_side.strip().lower()
        if side not in {"red", "black"}:
            raise ValueError("initial side must be red or black")
        if search_depth <= 0:
            raise ValueError("search depth must be positive")
        if move_time_ms is not None and move_time_ms <= 0:
            raise ValueError("move time must be positive")
        self.engine = engine
        self.side_to_move = side
        self.search_depth = search_depth
        self.move_time_ms = move_time_ms
        self.stability = StableBoardDetector(stable_frames)
        self.board: BoardState | None = None

    @staticmethod
    def _search_event_fields(
        analysis: ProtocolSearchResult,
    ) -> dict[str, str | int | None]:
        return {
            "best_move": analysis.best_move,
            "score": analysis.score,
            "depth": analysis.depth,
            "timed_out": analysis.timed_out,
        }

    def _current_event_fields(self) -> dict[str, str | int | None]:
        """查询明确裁决；对局继续时才启动搜索。"""

        result = self.engine.result()
        if result.finished:
            return {
                "best_move": None,
                "score": None,
                "depth": None,
                "outcome": result.outcome,
                "reason": result.reason,
            }
        return {
            **self._search_event_fields(
                self.engine.search(
                    self.search_depth, move_time_ms=self.move_time_ms
                )
            ),
            "outcome": result.outcome,
            "reason": result.reason,
        }

    def process_recognition(self, result: RecognitionResult) -> ObserverEvent | None:
        """提交单帧识别结果；尚未稳定时返回 None。"""

        return self.process_board(BoardState.from_recognition(result))

    def process_board(self, board: BoardState) -> ObserverEvent | None:
        """提交单帧棋盘；稳定后初始化或验证一步真实走法。"""

        stable = self.stability.update(board)
        if stable is None:
            return None
        if self.board is None:
            fen = stable.to_fen(self.side_to_move)
            state = self.engine.set_position(fen)
            self.board = stable
            return ObserverEvent(
                kind="initialized",
                fen=state.fen,
                **self._current_event_fields(),
            )

        previous = self.board
        try:
            detected = detect_move(
                previous,
                stable,
                self.side_to_move,
                self.engine.legal_moves(),
            )
            state = self.engine.play(detected.move)
            engine_board = BoardState.from_fen(state.fen)
            if engine_board != stable:
                self.engine.undo()
                raise ValueError("engine state does not match the recognized board")
        except (ValueError, RuntimeError) as error:
            self.stability.restore_accepted(previous)
            return ObserverEvent(kind="rejected", message=str(error))

        self.board = stable
        self.side_to_move = "black" if self.side_to_move == "red" else "red"
        return ObserverEvent(
            kind="move",
            fen=state.fen,
            move=detected.move,
            **self._current_event_fields(),
        )
