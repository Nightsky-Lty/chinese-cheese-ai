"""基于 mss 的跨平台窗口区域截图。"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Rect
from .windows import WindowInfo, activate_window


@dataclass(frozen=True)
class CapturedFrame:
    """一次窗口截图。

    参数:
        pixels: OpenCV 使用的 BGR 像素数组。
        screen_bounds: 截图在屏幕坐标中的原始范围。
    """

    pixels: np.ndarray
    screen_bounds: Rect

    @property
    def width(self) -> int:
        """返回截图的像素宽度。"""

        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """返回截图的像素高度。"""

        return int(self.pixels.shape[0])


def capture_rect(bounds: Rect) -> CapturedFrame:
    """截取指定屏幕矩形。

    参数:
        bounds: 使用操作系统全局坐标表示的截图区域。

    返回:
        BGR 格式图像以及其屏幕范围。Retina/DPI 下图像像素数可以与屏幕坐标不同。
    """

    try:
        import mss
    except ImportError as error:
        raise RuntimeError("screen capture requires the 'mss' package") from error

    monitor = {
        "left": round(bounds.left),
        "top": round(bounds.top),
        "width": round(bounds.width),
        "height": round(bounds.height),
    }
    with mss.mss() as screen:
        shot = screen.grab(monitor)
    pixels = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
    return CapturedFrame(pixels=pixels, screen_bounds=bounds)


def _capture_macos_window(window: WindowInfo) -> CapturedFrame:
    """通过窗口 ID 截取 macOS 窗口，即使窗口被其他窗口遮挡也可工作。"""

    try:
        import Quartz  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "macOS window capture requires pyobjc-framework-Quartz"
        ) from error

    options = (
        Quartz.kCGWindowImageBoundsIgnoreFraming
        | Quartz.kCGWindowImageBestResolution
    )
    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window.window_id,
        options,
    )
    if image is None:
        raise RuntimeError(
            "macOS could not capture the selected window; check Screen Recording permission"
        )

    width = int(Quartz.CGImageGetWidth(image))
    height = int(Quartz.CGImageGetHeight(image))
    bytes_per_row = int(Quartz.CGImageGetBytesPerRow(image))
    bits_per_pixel = int(Quartz.CGImageGetBitsPerPixel(image))
    if width <= 0 or height <= 0 or bits_per_pixel != 32:
        raise RuntimeError("macOS returned an unsupported window image format")

    provider = Quartz.CGImageGetDataProvider(image)
    data = Quartz.CGDataProviderCopyData(provider)
    raw = np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, bytes_per_row)
    bgra = raw[:, : width * 4].reshape(height, width, 4)
    return CapturedFrame(pixels=bgra[:, :, :3].copy(), screen_bounds=window.bounds)


def capture_window(window: WindowInfo) -> CapturedFrame:
    """截取指定窗口，而不是仅截取它原本所在的屏幕矩形。

    参数:
        window: 由窗口枚举器返回的目标窗口。

    返回:
        macOS 使用窗口 ID，可正确处理遮挡；Windows 先激活窗口，再进行区域截图。
    """

    if sys.platform == "darwin":
        return _capture_macos_window(window)
    if sys.platform == "win32":
        activate_window(window)
        time.sleep(0.2)
        return capture_rect(window.bounds)
    raise RuntimeError("GUI automation currently supports macOS and Windows only")


def save_frame(frame: CapturedFrame, output: str | Path) -> Path:
    """将截图保存为 PNG/JPEG 等 OpenCV 支持的格式。

    参数:
        frame: 要保存的截图。
        output: 输出文件路径，父目录不存在时自动创建。
    """

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("saving screenshots requires opencv-python") from error

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame.pixels):
        raise RuntimeError(f"failed to write screenshot: {path}")
    return path
