"""微信象棋界面的跨平台截图、棋盘标定与坐标控制工具。"""

from .geometry import BoardCalibration, Point, Rect, parse_move
from .recognition import RecognitionResult, TemplateLibrary, TemplatePieceRecognizer
from .windows import WindowInfo, find_windows, list_windows

__all__ = [
    "BoardCalibration",
    "Point",
    "Rect",
    "RecognitionResult",
    "TemplateLibrary",
    "TemplatePieceRecognizer",
    "WindowInfo",
    "find_windows",
    "list_windows",
    "parse_move",
]
