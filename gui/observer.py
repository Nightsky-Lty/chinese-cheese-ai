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
    moves: tuple[str, ...] = ()
    best_move: str | None = None
    score: int | None = None
    depth: int | None = None
    timed_out: bool = False
    outcome: str | None = None
    reason: str | None = None
    message: str | None = None
    changes: tuple[str, ...] = ()


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
        self._last_transient_board: BoardState | None = None

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

    def process_recognition(
        self,
        result: RecognitionResult,
        *,
        expected_move: str | None = None,
    ) -> ObserverEvent | None:
        """提交单帧识别结果；可用待确认走法追赶跨过的对手应手。

        参数:
            result: 当前截图的棋盘识别结果。
            expected_move: GUI 刚执行、但尚未被单独观察到的引擎走法。
        """

        return self.process_board(
            BoardState.from_recognition(result), expected_move=expected_move
        )

    def process_board(
        self,
        board: BoardState,
        *,
        expected_move: str | None = None,
    ) -> ObserverEvent | None:
        """提交单帧棋盘；稳定后验证一步，必要时追赶连续两步。

        当 ``expected_move`` 已由 GUI 点击，而首个稳定画面已经包含对手应手时，
        观察器会依次验证该预期走法和唯一合法应手，并原子地推进两层历史。
        """

        if self.board is not None and board == self.board:
            self._last_transient_board = None
        stable = self.stability.update(board)
        if stable is None:
            return None
        if self.board is None:
            if stable.has_unknown:
                self.stability.restore_accepted(None)
                return None
            fen = stable.to_fen(self.side_to_move)
            state = self.engine.set_position(fen)
            self.board = stable
            self._last_transient_board = None
            return ObserverEvent(
                kind="initialized",
                fen=state.fen,
                **self._current_event_fields(),
            )

        previous = self.board
        changes = self._describe_changes(previous, stable)
        if stable.has_unknown:
            if expected_move is not None:
                expected = self._recover_expected_move_from_partial(
                    previous, stable, expected_move
                )
                if expected is not None:
                    return expected
                try:
                    return self._recover_expected_move_and_reply(
                        previous, stable, expected_move
                    )
                except (ValueError, RuntimeError):
                    pass
            self.stability.restore_accepted(previous)
            if stable == self._last_transient_board:
                return None
            self._last_transient_board = stable
            detail = ", ".join(changes) if changes else "unknown squares only"
            return ObserverEvent(
                kind="transient",
                message=f"partial intermediate ignored: {detail}",
                changes=changes,
            )

        if len(changes) == 1:
            self.stability.restore_accepted(previous)
            if stable == self._last_transient_board:
                return None
            self._last_transient_board = stable
            return ObserverEvent(
                kind="transient",
                message=f"one-square intermediate ignored: {changes[0]}",
                changes=changes,
            )

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
            if expected_move is not None:
                try:
                    return self._recover_expected_move_and_reply(
                        previous, stable, expected_move
                    )
                except (ValueError, RuntimeError) as recovery_error:
                    error = ValueError(
                        f"{error}; two-ply recovery failed: {recovery_error}"
                    )
            self.stability.restore_accepted(previous)
            self._last_transient_board = None
            detail = ", ".join(changes) if changes else "no square changes"
            return ObserverEvent(
                kind="rejected",
                message=f"{error}; changes: {detail}",
                changes=changes,
            )

        self.board = stable
        self._last_transient_board = None
        self.side_to_move = "black" if self.side_to_move == "red" else "red"
        return ObserverEvent(
            kind="move",
            fen=state.fen,
            move=detected.move,
            **self._current_event_fields(),
        )

    def _recover_expected_move_and_reply(
        self,
        previous: BoardState,
        stable: BoardState,
        expected_move: str,
    ) -> ObserverEvent:
        """验证未单独观察到的预期走法及紧随其后的唯一合法应手。

        参数:
            previous: 最近一次已被引擎与视觉共同确认的棋盘。
            stable: 当前稳定棋盘，预期已经包含连续两个半回合。
            expected_move: GUI 已发送、正在等待确认的第一步。

        任一步验证失败时会撤销本方法写入的全部引擎历史，保证恢复过程原子化。
        """

        legal_moves = set(self.engine.legal_moves())
        if expected_move not in legal_moves:
            raise ValueError(f"expected move {expected_move} is not legal")

        played = 0
        try:
            first_state = self.engine.play(expected_move)
            played = 1
            intermediate = BoardState.from_fen(first_state.fen)
            expected_board = self._apply_move(previous, expected_move)
            if intermediate != expected_board:
                raise ValueError("engine state does not match the expected GUI move")

            reply_side = "black" if self.side_to_move == "red" else "red"
            if stable.has_unknown:
                reply_move = self._unique_partial_reply(
                    previous,
                    intermediate,
                    stable,
                    self.engine.legal_moves(),
                )
            else:
                reply_move = detect_move(
                    intermediate,
                    stable,
                    reply_side,
                    self.engine.legal_moves(),
                ).move
            final_state = self.engine.play(reply_move)
            played = 2
            final_board = BoardState.from_fen(final_state.fen)
            if not self._matches_observation(final_board, stable):
                raise ValueError("engine state does not match the recovered board")
        except (ValueError, RuntimeError):
            for _ in range(played):
                self.engine.undo()
            raise

        self.board = final_board
        self.stability.restore_accepted(final_board)
        self._last_transient_board = None
        # 连续推进两个半回合后，行棋方与恢复前相同。
        return ObserverEvent(
            kind="moves",
            fen=final_state.fen,
            move=reply_move,
            moves=(expected_move, reply_move),
            **self._current_event_fields(),
        )

    def _recover_expected_move_from_partial(
        self,
        previous: BoardState,
        observed: BoardState,
        expected_move: str,
    ) -> ObserverEvent | None:
        """在起点和终点均已知时，从部分视觉局面确认待执行的单步走法。"""

        if expected_move not in set(self.engine.legal_moves()):
            return None
        expected_board = self._apply_move(previous, expected_move)
        changed_indices = [
            index
            for index, (before, after) in enumerate(
                zip(previous.pieces, expected_board.pieces)
            )
            if before != after
        ]
        if not all(observed.pieces[index] != "?" for index in changed_indices):
            return None
        if not self._matches_observation(expected_board, observed):
            return None

        state = self.engine.play(expected_move)
        engine_board = BoardState.from_fen(state.fen)
        if engine_board != expected_board:
            self.engine.undo()
            raise ValueError("engine state does not match the expected GUI move")
        self.board = engine_board
        self.stability.restore_accepted(engine_board)
        self._last_transient_board = None
        self.side_to_move = "black" if self.side_to_move == "red" else "red"
        return ObserverEvent(
            kind="move",
            fen=state.fen,
            move=expected_move,
            **self._current_event_fields(),
        )

    def _unique_partial_reply(
        self,
        previous: BoardState,
        intermediate: BoardState,
        observed: BoardState,
        legal_replies: tuple[str, ...] | set[str],
    ) -> str:
        """返回与部分视觉棋盘一致的唯一合法应手。

        至少要求最终局面相对可信局面有两个已知变化格，防止仅凭一个动画格猜测。
        """

        candidates: list[str] = []
        for move in legal_replies:
            final_board = self._apply_move(intermediate, move)
            if not self._matches_observation(final_board, observed):
                continue
            evidence = sum(
                seen != "?" and before != after
                for before, after, seen in zip(
                    previous.pieces, final_board.pieces, observed.pieces
                )
            )
            if evidence >= 2:
                candidates.append(move)
        if len(candidates) != 1:
            raise ValueError(
                "partial board does not identify exactly one legal reply "
                f"(candidates={len(candidates)})"
            )
        return candidates[0]

    @staticmethod
    def _matches_observation(expected: BoardState, observed: BoardState) -> bool:
        """未知格作为通配符时，完整引擎棋盘是否与视觉观察一致。"""

        return all(
            seen == "?" or actual == seen
            for actual, seen in zip(expected.pieces, observed.pieces)
        )

    @staticmethod
    def _describe_changes(
        previous: BoardState, current: BoardState
    ) -> tuple[str, ...]:
        """返回形如 ``i9 r->.`` 的逐格变化描述，供过滤和诊断日志使用。"""

        descriptions: list[str] = []
        for index, (before, after) in enumerate(
            zip(previous.pieces, current.pieces)
        ):
            if before == after:
                continue
            square = chr(ord("a") + index % 9) + str(index // 9)
            descriptions.append(f"{square} {before}->{after}")
        return tuple(descriptions)

    @staticmethod
    def _apply_move(board: BoardState, move: str) -> BoardState:
        """在纯棋盘上应用四字符坐标走法，用于交叉验证引擎中间局面。"""

        if len(move) != 4:
            raise ValueError(f"invalid coordinate move: {move}")
        source_file = ord(move[0]) - ord("a")
        source_rank = ord(move[1]) - ord("0")
        target_file = ord(move[2]) - ord("a")
        target_rank = ord(move[3]) - ord("0")
        if not (
            0 <= source_file < 9
            and 0 <= target_file < 9
            and 0 <= source_rank < 10
            and 0 <= target_rank < 10
        ):
            raise ValueError(f"invalid coordinate move: {move}")
        source = source_rank * 9 + source_file
        target = target_rank * 9 + target_file
        pieces = list(board.pieces)
        if pieces[source] == ".":
            raise ValueError(f"expected move {move} starts from an empty square")
        pieces[target] = pieces[source]
        pieces[source] = "."
        return BoardState(tuple(pieces))
