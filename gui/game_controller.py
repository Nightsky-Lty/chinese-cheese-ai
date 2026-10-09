"""安全地编排棋盘观察、引擎建议与单步走法执行。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .board_state import BoardState
from .observer import GameObserver, ObserverEvent
from .recognition import RecognitionResult


class ControllerPhase(str, Enum):
    """连续对局控制器当前所处的阶段。"""

    WAITING_BOARD = "waiting_board"
    WAITING_OPPONENT = "waiting_opponent"
    WAITING_RECOVERY_SETTLE = "waiting_recovery_settle"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    PAUSED = "paused"
    FINISHED = "finished"


class PauseKind(str, Enum):
    """暂停原因分类，用于限制可以安全执行的人工恢复动作。"""

    RECOGNITION_REJECTED = "recognition_rejected"
    CONFIRMATION_TIMEOUT = "confirmation_timeout"
    MOVE_MISMATCH = "move_mismatch"
    EXECUTOR_FAILED = "executor_failed"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ControllerEvent:
    """控制器处理一帧棋盘后产生的可观察事件。

    参数:
        kind: 事件类型，例如 ``move_requested``、``move_confirmed`` 或 ``paused``。
        phase: 事件发生后控制器所处的阶段。
        move: 与事件相关的四字符引擎走法。
        message: 面向日志或界面的补充说明。
        observer_event: 触发本事件的底层观察器事件。
        pause_kind: 暂停事件的机器可读原因分类。
    """

    kind: str
    phase: ControllerPhase
    move: str | None = None
    message: str | None = None
    observer_event: ObserverEvent | None = None
    pause_kind: PauseKind | None = None


MoveExecutor = Callable[[str], None]


class GameController:
    """驱动一局连续对弈，并在每次执行后等待视觉确认。

    该类本身不包含鼠标实现。调用者必须显式传入 ``move_executor``；因此可以在
    离线测试中使用只记录走法的函数，未来也可以接入受控的 GUI 点击器。
    """

    def __init__(
        self,
        observer: GameObserver,
        *,
        ai_side: str,
        move_executor: MoveExecutor,
        confirmation_frame_limit: int = 30,
        recovery_settle_frames: int = 0,
    ) -> None:
        """创建连续对局控制器。

        参数:
            observer: 负责稳定帧、合法着差分及 C++ 历史同步的观察器。
            ai_side: 由引擎控制的一方，只能是 ``red`` 或 ``black``。
            move_executor: 接收四字符走法的单步执行函数；异常会令控制器暂停。
            confirmation_frame_limit: 请求落子后允许等待的最大识别帧数。
            recovery_settle_frames: 双步恢复后，再次点击前要求的完整可信画面帧数。
        """

        normalized_side = ai_side.strip().lower()
        if normalized_side not in {"red", "black"}:
            raise ValueError("AI side must be red or black")
        if confirmation_frame_limit <= 0:
            raise ValueError("confirmation frame limit must be positive")
        if recovery_settle_frames < 0:
            raise ValueError("recovery settle frame count must not be negative")
        self.observer = observer
        self.ai_side = normalized_side
        self.move_executor = move_executor
        self.confirmation_frame_limit = confirmation_frame_limit
        self.recovery_settle_frames = recovery_settle_frames
        self.phase = ControllerPhase.WAITING_BOARD
        self.pending_move: str | None = None
        self.pause_reason: str | None = None
        self._confirmation_frames = 0
        self._pause_kind: PauseKind | None = None
        self._paused_observer_event: ObserverEvent | None = None
        self._settling_move: str | None = None
        self._settling_event: ObserverEvent | None = None
        self._settled_frames = 0

    @property
    def active(self) -> bool:
        """控制器仍可接收棋盘帧时返回 True。"""

        return self.phase not in {ControllerPhase.PAUSED, ControllerPhase.FINISHED}

    @property
    def can_resume_waiting(self) -> bool:
        """当前暂停是否允许在不改变引擎历史的情况下继续观察。"""

        return self.phase == ControllerPhase.PAUSED and self._pause_kind in {
            PauseKind.RECOGNITION_REJECTED,
            PauseKind.CONFIRMATION_TIMEOUT,
            PauseKind.EXTERNAL,
        }

    @property
    def can_accept_observed_move(self) -> bool:
        """是否存在已通过 C++ 合法性检查、但与预期不同的实际走法。"""

        event = self._paused_observer_event
        return (
            self.phase == ControllerPhase.PAUSED
            and self._pause_kind == PauseKind.MOVE_MISMATCH
            and event is not None
            and event.kind == "move"
            and event.move is not None
        )

    def pause(
        self, reason: str, kind: PauseKind = PauseKind.EXTERNAL
    ) -> ControllerEvent:
        """安全暂停控制器，并保存需要人工检查的原因。"""

        self.phase = ControllerPhase.PAUSED
        self.pause_reason = reason
        self._pause_kind = kind
        self._paused_observer_event = None
        return ControllerEvent(
            "paused", self.phase, self.pending_move, reason, pause_kind=kind
        )

    def resume_waiting(self) -> ControllerEvent:
        """人工确认局面安全后继续观察，但绝不重新发送鼠标点击。"""

        if not self.can_resume_waiting:
            raise RuntimeError("this pause cannot safely resume waiting")
        self._confirmation_frames = 0
        self.pause_reason = None
        self._pause_kind = None
        self._paused_observer_event = None
        if self.pending_move is not None:
            self.phase = ControllerPhase.AWAITING_CONFIRMATION
        elif self.observer.board is None:
            self.phase = ControllerPhase.WAITING_BOARD
        else:
            self.phase = ControllerPhase.WAITING_OPPONENT
        return ControllerEvent(
            "resumed",
            self.phase,
            self.pending_move,
            "continued without repeating the previous click",
        )

    def accept_observed_move(self) -> tuple[ControllerEvent, ...]:
        """人工接受与预期不同、但已由 C++ 验证合法的实际走法。"""

        if not self.can_accept_observed_move:
            raise RuntimeError("there is no validated observed move to accept")
        event = self._paused_observer_event
        assert event is not None
        expected = self.pending_move
        observed = event.move
        self.pending_move = None
        self._confirmation_frames = 0
        self.pause_reason = None
        self._pause_kind = None
        self._paused_observer_event = None
        following = self._advance_after_event(event, [])
        recovered = ControllerEvent(
            "observed_move_accepted",
            self.phase,
            observed,
            f"accepted {observed} instead of expected {expected}",
            event,
        )
        return (recovered, *following)

    def process_recognition(self, result: RecognitionResult) -> tuple[ControllerEvent, ...]:
        """处理一帧识别结果，并返回本帧产生的全部控制事件。"""

        if not self.active:
            return ()
        board = BoardState.from_recognition(result)
        return self._process_event(
            self.observer.process_recognition(
                result, expected_move=self.pending_move
            ),
            board,
        )

    def process_board(self, board: BoardState) -> tuple[ControllerEvent, ...]:
        """处理一帧测试棋盘，并返回本帧产生的全部控制事件。"""

        if not self.active:
            return ()
        return self._process_event(
            self.observer.process_board(board, expected_move=self.pending_move),
            board,
        )

    def _process_event(
        self, event: ObserverEvent | None, board: BoardState
    ) -> tuple[ControllerEvent, ...]:
        """处理观察结果，并对未确认的执行走法实施帧数超时。"""

        if self.phase == ControllerPhase.WAITING_RECOVERY_SETTLE:
            if event is None:
                return self._settle_recovered_board(board)
            self._settled_frames = 0
            if event.kind != "transient":
                self._settling_move = None
                self._settling_event = None
        if event is not None:
            emitted = self._handle_observer_event(event)
            if event.kind == "transient" and self.pending_move is not None:
                return (*emitted, *self._tick_confirmation())
            return emitted
        if self.pending_move is None:
            return ()
        return self._tick_confirmation()

    def _tick_confirmation(self) -> tuple[ControllerEvent, ...]:
        """每张待确认画面都计入上限，包括产生 transient 日志的画面。"""

        assert self.pending_move is not None
        self._confirmation_frames += 1
        if self._confirmation_frames < self.confirmation_frame_limit:
            return ()
        reason = (
            f"move {self.pending_move} was not confirmed within "
            f"{self.confirmation_frame_limit} frames"
        )
        self.phase = ControllerPhase.PAUSED
        self.pause_reason = reason
        self._pause_kind = PauseKind.CONFIRMATION_TIMEOUT
        self._paused_observer_event = None
        return (
            ControllerEvent(
                "paused",
                self.phase,
                self.pending_move,
                reason,
                pause_kind=self._pause_kind,
            ),
        )

    def _settle_recovered_board(self, board: BoardState) -> tuple[ControllerEvent, ...]:
        """双步恢复后，等完整可信棋盘连续出现再发下一次点击。"""

        if board.has_unknown or board != self.observer.board:
            self._settled_frames = 0
            return ()
        self._settled_frames += 1
        if self._settled_frames < self.recovery_settle_frames:
            return ()
        move = self._settling_move
        event = self._settling_event
        assert move is not None and event is not None
        self._settling_move = None
        self._settling_event = None
        self._settled_frames = 0
        return self._request_move(move, event, [])

    def _handle_observer_event(
        self, event: ObserverEvent | None
    ) -> tuple[ControllerEvent, ...]:
        """把观察器事件转换为安全的连续对局状态变化。"""

        if event is None:
            return ()
        if event.kind == "transient":
            return (
                ControllerEvent(
                    "transient",
                    self.phase,
                    self.pending_move,
                    event.message,
                    event,
                ),
            )
        if event.kind == "rejected":
            reason = event.message or "recognized board was rejected"
            self.phase = ControllerPhase.PAUSED
            self.pause_reason = reason
            self._pause_kind = PauseKind.RECOGNITION_REJECTED
            self._paused_observer_event = event
            return (
                ControllerEvent(
                    "paused",
                    self.phase,
                    self.pending_move,
                    reason,
                    event,
                    self._pause_kind,
                ),
            )

        emitted: list[ControllerEvent] = []
        if event.kind == "moves":
            moves = event.moves
            if (
                self.pending_move is None
                or len(moves) != 2
                or moves[0] != self.pending_move
            ):
                reason = "two-ply recovery does not match the pending GUI move"
                self.phase = ControllerPhase.PAUSED
                self.pause_reason = reason
                self._pause_kind = PauseKind.MOVE_MISMATCH
                self._paused_observer_event = event
                return (
                    ControllerEvent(
                        "paused",
                        self.phase,
                        self.pending_move,
                        reason,
                        event,
                        self._pause_kind,
                    ),
                )
            confirmed, reply = moves
            self.pending_move = None
            self._confirmation_frames = 0
            emitted.extend(
                (
                    ControllerEvent(
                        "move_confirmed",
                        self.phase,
                        confirmed,
                        observer_event=event,
                    ),
                    ControllerEvent(
                        "opponent_move",
                        self.phase,
                        reply,
                        "recovered together with the preceding AI move",
                        event,
                    ),
                )
            )
        elif event.kind == "reply_corrected":
            if self.pending_move is not None or len(event.moves) != 2:
                reason = "recovered reply changed while another move was pending"
                self.phase = ControllerPhase.PAUSED
                self.pause_reason = reason
                self._pause_kind = PauseKind.MOVE_MISMATCH
                self._paused_observer_event = event
                return (
                    ControllerEvent(
                        "paused",
                        self.phase,
                        self.pending_move,
                        reason,
                        event,
                        self._pause_kind,
                    ),
                )
            emitted.append(
                ControllerEvent(
                    "opponent_move_corrected",
                    self.phase,
                    event.move,
                    event.message,
                    event,
                )
            )
        elif event.kind == "move":
            if self.pending_move is not None:
                if event.move != self.pending_move:
                    reason = (
                        f"executed move {self.pending_move} but observed "
                        f"{event.move or 'no move'}"
                    )
                    self.phase = ControllerPhase.PAUSED
                    self.pause_reason = reason
                    self._pause_kind = PauseKind.MOVE_MISMATCH
                    self._paused_observer_event = event
                    return (
                        ControllerEvent(
                            "paused",
                            self.phase,
                            self.pending_move,
                            reason,
                            event,
                            self._pause_kind,
                        ),
                    )
                confirmed = self.pending_move
                self.pending_move = None
                self._confirmation_frames = 0
                emitted.append(
                    ControllerEvent(
                        "move_confirmed",
                        self.phase,
                        confirmed,
                        observer_event=event,
                    )
                )
            else:
                emitted.append(
                    ControllerEvent(
                        "opponent_move",
                        self.phase,
                        event.move,
                        observer_event=event,
                    )
                )
        elif event.kind == "initialized":
            emitted.append(
                ControllerEvent("initialized", self.phase, observer_event=event)
            )

        return self._advance_after_event(event, emitted)

    def _advance_after_event(
        self, event: ObserverEvent, emitted: list[ControllerEvent]
    ) -> tuple[ControllerEvent, ...]:
        """根据已接受局面的行棋方继续等待、结束或请求唯一一步执行。"""

        if event.best_move is None:
            self.phase = ControllerPhase.FINISHED
            if event.outcome and event.outcome != "ongoing":
                result_message = f"{event.outcome}: {event.reason or 'unknown'}"
            else:
                result_message = "engine returned no legal move"
            emitted.append(
                ControllerEvent(
                    "finished",
                    self.phase,
                    message=result_message,
                    observer_event=event,
                )
            )
            return tuple(emitted)

        if self.observer.side_to_move != self.ai_side:
            self.phase = ControllerPhase.WAITING_OPPONENT
            return tuple(emitted)

        if self.pending_move is not None:
            self.phase = ControllerPhase.AWAITING_CONFIRMATION
            return tuple(emitted)

        if event.kind in {"moves", "reply_corrected"} and self.recovery_settle_frames:
            self.phase = ControllerPhase.WAITING_RECOVERY_SETTLE
            self._settling_move = event.best_move
            self._settling_event = event
            self._settled_frames = 0
            return tuple(emitted)

        return self._request_move(event.best_move, event, emitted)

    def _request_move(
        self, move: str, event: ObserverEvent, emitted: list[ControllerEvent]
    ) -> tuple[ControllerEvent, ...]:
        """发送一次待确认走法，失败时保持暂停且不自动重试。"""

        try:
            self.move_executor(move)
        except Exception as error:
            reason = f"move executor failed for {move}: {error}"
            self.phase = ControllerPhase.PAUSED
            self.pause_reason = reason
            self._pause_kind = PauseKind.EXECUTOR_FAILED
            self._paused_observer_event = event
            emitted.append(
                ControllerEvent(
                    "paused",
                    self.phase,
                    move,
                    reason,
                    event,
                    self._pause_kind,
                )
            )
            return tuple(emitted)

        self.pending_move = move
        self._confirmation_frames = 0
        self.phase = ControllerPhase.AWAITING_CONFIRMATION
        emitted.append(
            ControllerEvent(
                "move_requested",
                self.phase,
                self.pending_move,
                observer_event=event,
            )
        )
        return tuple(emitted)
