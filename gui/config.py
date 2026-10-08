"""棋盘标定 JSON 的读取与保存。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .geometry import BoardCalibration, Point


@dataclass(frozen=True)
class GuiConfig:
    """重新定位微信窗口和换算棋盘坐标所需的配置。

    参数:
        window_query: 应用名或窗口标题的搜索文本。
        calibration: 棋盘四角和朝向标定。
    """

    window_query: str
    calibration: BoardCalibration


def save_config(config: GuiConfig, path: str | Path) -> Path:
    """保存版本化的棋盘标定配置。

    参数:
        config: 要保存的 GUI 配置。
        path: JSON 输出路径。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "window_query": config.window_query,
        "red_at_bottom": config.calibration.red_at_bottom,
        "corners": [
            {"x": point.x, "y": point.y} for point in config.calibration.corners
        ],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def load_config(path: str | Path) -> GuiConfig:
    """读取并校验棋盘标定配置。

    参数:
        path: 由 :func:`save_config` 生成的 JSON 文件。
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported GUI configuration version")
    raw_corners = payload.get("corners")
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        raise ValueError("GUI configuration must contain four board corners")
    corners = tuple(Point(float(item["x"]), float(item["y"])) for item in raw_corners)
    return GuiConfig(
        window_query=str(payload.get("window_query", "")),
        calibration=BoardCalibration(
            corners=corners,  # type: ignore[arg-type]
            red_at_bottom=bool(payload.get("red_at_bottom", True)),
        ),
    )
