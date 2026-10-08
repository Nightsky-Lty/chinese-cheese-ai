"""鼠标预览和受控点击。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .geometry import BoardCalibration, Point
from .windows import WindowInfo, activate_window


@dataclass(frozen=True)
class MoveExecution:
    """一次走法坐标换算或点击结果。

    参数:
        source: 起点屏幕坐标。
        target: 终点屏幕坐标。
        executed: 是否真的发送了鼠标点击。
    """

    source: Point
    target: Point
    executed: bool


def execute_move(
    window: WindowInfo,
    calibration: BoardCalibration,
    move: str,
    *,
    execute: bool = False,
    click_interval: float = 0.25,
) -> MoveExecution:
    """预览或点击执行一个象棋走法。

    参数:
        window: 当前目标微信小程序窗口。
        calibration: 与该界面对应的棋盘标定。
        move: 形如 ``b0c2`` 的引擎走法。
        execute: 默认 ``False``，只返回坐标；显式设为 ``True`` 才实际点击。
        click_interval: 点击起点和终点之间的等待秒数。
    """

    if click_interval < 0:
        raise ValueError("click interval cannot be negative")
    source, target = calibration.move_screen_points(move, window.bounds)
    if not window.bounds.contains(source) or not window.bounds.contains(target):
        raise RuntimeError("calibrated move falls outside the selected window")
    if not execute:
        return MoveExecution(source, target, False)

    try:
        import pyautogui
    except ImportError as error:
        raise RuntimeError("mouse control requires the 'pyautogui' package") from error

    activate_window(window)
    time.sleep(0.2)
    pyautogui.click(round(source.x), round(source.y))
    time.sleep(click_interval)
    pyautogui.click(round(target.x), round(target.y))
    return MoveExecution(source, target, True)
