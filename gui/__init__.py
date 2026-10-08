"""微信象棋界面的跨平台截图、棋盘标定与坐标控制工具。"""

from .geometry import BoardCalibration, Point, Rect, parse_move
from .board_state import BoardState, DetectedMove, StableBoardDetector, detect_move
from .game_controller import ControllerEvent, ControllerPhase, GameController, PauseKind
from .recognition import RecognitionResult, TemplateLibrary, TemplatePieceRecognizer
from .observer import GameObserver, ObserverEvent
from .protocol_client import ProtocolGameResult
from .windows import WindowInfo, find_windows, list_windows

__all__ = [
    "BoardCalibration",
    "BoardState",
    "ControllerEvent",
    "ControllerPhase",
    "DetectedMove",
    "GameObserver",
    "GameController",
    "PauseKind",
    "Point",
    "ObserverEvent",
    "ProtocolGameResult",
    "Rect",
    "RecognitionResult",
    "TemplateLibrary",
    "TemplatePieceRecognizer",
    "StableBoardDetector",
    "WindowInfo",
    "find_windows",
    "detect_move",
    "list_windows",
    "parse_move",
]
