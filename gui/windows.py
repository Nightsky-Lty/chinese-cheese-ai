"""macOS 和 Windows 的顶层窗口枚举与激活。"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .geometry import Rect


@dataclass(frozen=True)
class WindowInfo:
    """一个可见顶层窗口的信息。

    参数:
        window_id: 操作系统窗口编号。
        owner: 创建窗口的应用名称。
        title: 窗口标题。
        bounds: 窗口在全局屏幕坐标中的范围。
        process_id: 所属进程编号，无法取得时为 0。
    """

    window_id: int
    owner: str
    title: str
    bounds: Rect
    process_id: int = 0

    @property
    def label(self) -> str:
        """返回适合命令行显示的窗口名称。"""

        if self.owner and self.title:
            return f"{self.owner} - {self.title}"
        return self.title or self.owner or f"window {self.window_id}"


def _list_macos_windows() -> list[WindowInfo]:
    """使用 Quartz 枚举 macOS 当前屏幕上的普通窗口。"""

    try:
        import Quartz  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "macOS window discovery requires pyobjc-framework-Quartz"
        ) from error

    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    raw_windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    if raw_windows is None:
        raise RuntimeError(
            "macOS did not return a window list; grant Screen Recording permission "
            "to the terminal/IDE running Python, then restart that application"
        )
    result: list[WindowInfo] = []
    for raw in raw_windows:
        bounds = raw.get(Quartz.kCGWindowBounds, {})
        width = float(bounds.get("Width", 0))
        height = float(bounds.get("Height", 0))
        layer = int(raw.get(Quartz.kCGWindowLayer, 0))
        alpha = float(raw.get(Quartz.kCGWindowAlpha, 1.0))
        if layer != 0 or alpha <= 0 or width < 80 or height < 80:
            continue
        result.append(
            WindowInfo(
                window_id=int(raw.get(Quartz.kCGWindowNumber, 0)),
                owner=str(raw.get(Quartz.kCGWindowOwnerName, "")),
                title=str(raw.get(Quartz.kCGWindowName, "")),
                bounds=Rect(
                    float(bounds.get("X", 0)),
                    float(bounds.get("Y", 0)),
                    width,
                    height,
                ),
                process_id=int(raw.get(Quartz.kCGWindowOwnerPID, 0)),
            )
        )
    return result


def _list_windows_windows() -> list[WindowInfo]:
    """使用 Win32 API 枚举 Windows 当前可见的顶层窗口。"""

    try:
        import win32gui  # type: ignore[import-not-found]
        import win32process  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Windows window discovery requires pywin32") from error

    result: list[WindowInfo] = []

    def collect(handle: int, _: object) -> None:
        if not win32gui.IsWindowVisible(handle):
            return
        title = win32gui.GetWindowText(handle).strip()
        if not title:
            return
        left, top, right, bottom = win32gui.GetWindowRect(handle)
        width = right - left
        height = bottom - top
        if width < 80 or height < 80:
            return
        _, process_id = win32process.GetWindowThreadProcessId(handle)
        result.append(
            WindowInfo(
                window_id=int(handle),
                owner="",
                title=title,
                bounds=Rect(float(left), float(top), float(width), float(height)),
                process_id=int(process_id),
            )
        )

    win32gui.EnumWindows(collect, None)
    return result


def list_windows() -> list[WindowInfo]:
    """列出当前系统可用于截图的可见顶层窗口。"""

    if sys.platform == "darwin":
        return _list_macos_windows()
    if sys.platform == "win32":
        return _list_windows_windows()
    raise RuntimeError("GUI automation currently supports macOS and Windows only")


def find_windows(query: str) -> list[WindowInfo]:
    """按应用名或窗口标题查找窗口，不区分大小写。

    参数:
        query: 应用名或窗口标题包含的文本；空字符串匹配所有窗口。
    """

    needle = query.casefold().strip()
    windows = list_windows()
    if not needle:
        return windows
    return [window for window in windows if needle in window.label.casefold()]


def activate_window(window: WindowInfo) -> None:
    """将指定窗口所属应用切换到前台。

    参数:
        window: 要激活的窗口。macOS 以进程为单位激活，Windows 激活具体窗口。
    """

    if sys.platform == "darwin":
        try:
            from AppKit import (  # type: ignore[import-not-found]
                NSApplicationActivateIgnoringOtherApps,
                NSRunningApplication,
            )
            import ApplicationServices as AS  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "macOS window activation requires pyobjc Cocoa and ApplicationServices"
            ) from error
        application = NSRunningApplication.runningApplicationWithProcessIdentifier_(
            window.process_id
        )
        if application is None:
            raise RuntimeError("the selected application is no longer running")
        application.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

        app_element = AS.AXUIElementCreateApplication(window.process_id)
        error_code, app_windows = AS.AXUIElementCopyAttributeValue(
            app_element, AS.kAXWindowsAttribute, None
        )
        if error_code != 0 or app_windows is None:
            raise RuntimeError(
                "cannot inspect application windows; grant Accessibility permission "
                "to the terminal/IDE running Python"
            )

        def attribute(element: object, name: str) -> object | None:
            """读取一个辅助功能属性，读取失败时返回 None。"""

            code, value = AS.AXUIElementCopyAttributeValue(element, name, None)
            return value if code == 0 else None

        target = None
        for candidate in app_windows:
            title = attribute(candidate, AS.kAXTitleAttribute)
            position_value = attribute(candidate, AS.kAXPositionAttribute)
            size_value = attribute(candidate, AS.kAXSizeAttribute)
            if position_value is None or size_value is None:
                continue
            position_ok, position = AS.AXValueGetValue(
                position_value, AS.kAXValueCGPointType, None
            )
            size_ok, size = AS.AXValueGetValue(size_value, AS.kAXValueCGSizeType, None)
            if not position_ok or not size_ok:
                continue
            same_title = str(title or "") == window.title
            same_bounds = (
                abs(float(position.x) - window.bounds.left) <= 2
                and abs(float(position.y) - window.bounds.top) <= 2
                and abs(float(size.width) - window.bounds.width) <= 2
                and abs(float(size.height) - window.bounds.height) <= 2
            )
            if same_title and same_bounds:
                target = candidate
                break
        if target is None:
            raise RuntimeError("the selected macOS window is no longer available")
        if AS.AXUIElementPerformAction(target, AS.kAXRaiseAction) != 0:
            raise RuntimeError("macOS could not raise the selected window")
        return

    if sys.platform == "win32":
        try:
            import win32con  # type: ignore[import-not-found]
            import win32gui  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Windows window activation requires pywin32") from error
        win32gui.ShowWindow(window.window_id, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window.window_id)
        return

    raise RuntimeError("GUI automation currently supports macOS and Windows only")
