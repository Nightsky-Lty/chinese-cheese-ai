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
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass(frozen=True)
class ControllerEvent:
    """控制器处理一帧棋盘后产生的可观察事件。

    参数:
        kind: 事件类型，例如 ``move_requested``、``move_confirmed`` 或 ``paused``。
        phase: 事件发生后控制器所处的阶段。
        move: 与事件相关的四字符引擎走法。
        message: 面向日志或界面的补充说明。
        observer_event: 触发本事件的底层观察器事件。
    """

    kind: str
    phase: ControllerPhase
    move: str | None = None
    message: str | None = None
    observer_event: ObserverEvent | None = None


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
    ) -> None:
        """创建连续对局控制器。

        参数:
            observer: 负责稳定帧、合法着差分及 C++ 历史同步的观察器。
            ai_side: 由引擎控制的一方，只能是 ``red`` 或 ``black``。
            move_executor: 接收四字符走法的单步执行函数；异常会令控制器暂停。
            confirmation_frame_limit: 请求落子后允许等待的最大识别帧数。
        """

        normalized_side = ai_side.strip().lower()
        if normalized_side not in {"red", "black"}:
            raise ValueError("AI side must be red or black")
        if confirmation_frame_limit <= 0:
            raise ValueError("confirmation frame limit must be positive")
        self.observer = observer
        self.ai_side = normalized_side
        self.move_executor = move_executor
        self.confirmation_frame_limit = confirmation_frame_limit
        self.phase = ControllerPhase.WAITING_BOARD
        self.pending_move: str | None = None
        self.pause_reason: str | None = None
        self._confirmation_frames = 0

    @property
    def active(self) -> bool:
        """控制器仍可接收棋盘帧时返回 True。"""

        return self.phase not in {ControllerPhase.PAUSED, ControllerPhase.FINISHED}

    def pause(self, reason: str) -> ControllerEvent:
        """安全暂停控制器，并保存需要人工检查的原因。"""

        self.phase = ControllerPhase.PAUSED
        self.pause_reason = reason
        return ControllerEvent("paused", self.phase, self.pending_move, reason)

    def process_recognition(self, result: RecognitionResult) -> tuple[ControllerEvent, ...]:
        """处理一帧识别结果，并返回本帧产生的全部控制事件。"""

        if not self.active:
            return ()
        return self._process_event(self.observer.process_recognition(result))

    def process_board(self, board: BoardState) -> tuple[ControllerEvent, ...]:
        """处理一帧测试棋盘，并返回本帧产生的全部控制事件。"""

        if not self.active:
            return ()
        return self._process_event(self.observer.process_board(board))

    def _process_event(
        self, event: ObserverEvent | None
    ) -> tuple[ControllerEvent, ...]:
        """处理观察结果，并对未确认的执行走法实施帧数超时。"""

        if event is not None:
            return self._handle_observer_event(event)
        if self.pending_move is None:
            return ()
        self._confirmation_frames += 1
        if self._confirmation_frames < self.confirmation_frame_limit:
            return ()
        reason = (
            f"move {self.pending_move} was not confirmed within "
            f"{self.confirmation_frame_limit} frames"
        )
        self.phase = ControllerPhase.PAUSED
        self.pause_reason = reason
        return (ControllerEvent("paused", self.phase, self.pending_move, reason),)

    def _handle_observer_event(
        self, event: ObserverEvent | None
    ) -> tuple[ControllerEvent, ...]:
        """把观察器事件转换为安全的连续对局状态变化。"""

        if event is None:
            return ()
        if event.kind == "rejected":
            reason = event.message or "recognized board was rejected"
            self.phase = ControllerPhase.PAUSED
            self.pause_reason = reason
            return (
                ControllerEvent(
                    "paused", self.phase, self.pending_move, reason, event
                ),
            )

        emitted: list[ControllerEvent] = []
        if event.kind == "move":
            if self.pending_move is not None:
                if event.move != self.pending_move:
                    reason = (
                        f"executed move {self.pending_move} but observed "
                        f"{event.move or 'no move'}"
                    )
                    self.phase = ControllerPhase.PAUSED
                    self.pause_reason = reason
                    return (
                        ControllerEvent(
                            "paused",
                            self.phase,
                            self.pending_move,
                            reason,
                            event,
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

        try:
            self.move_executor(event.best_move)
        except Exception as error:
            reason = f"move executor failed for {event.best_move}: {error}"
            self.phase = ControllerPhase.PAUSED
            self.pause_reason = reason
            emitted.append(
                ControllerEvent(
                    "paused", self.phase, event.best_move, reason, event
                )
            )
            return tuple(emitted)

        self.pending_move = event.best_move
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
