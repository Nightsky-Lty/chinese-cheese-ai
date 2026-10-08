"""与操作系统无关的窗口和棋盘坐标换算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Point:
    """二维坐标。

    参数:
        x: 横坐标。
        y: 纵坐标。
    """

    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    """屏幕上的矩形区域，坐标原点位于主屏幕左上角。

    参数:
        left: 矩形左边界的屏幕坐标。
        top: 矩形上边界的屏幕坐标。
        width: 矩形宽度。
        height: 矩形高度。
    """

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle width and height must be positive")

    def contains(self, point: Point) -> bool:
        """判断屏幕点是否位于矩形内（包含边界）。

        参数:
            point: 待检查的屏幕坐标。
        """

        return (
            self.left <= point.x <= self.left + self.width
            and self.top <= point.y <= self.top + self.height
        )


def parse_move(move: str) -> tuple[int, int, int, int]:
    """解析引擎使用的四字符坐标走法。

    参数:
        move: 形如 ``b0c2`` 的走法，列为 a～i，行为 0～9；0 为红方底线。

    返回:
        ``(起点列, 起点行, 终点列, 终点行)``。
    """

    text = move.strip().lower()
    if (
        len(text) != 4
        or text[0] < "a"
        or text[0] > "i"
        or text[2] < "a"
        or text[2] > "i"
        or not text[1].isdigit()
        or not text[3].isdigit()
    ):
        raise ValueError("move must look like b0c2 (files a-i, ranks 0-9)")
    return ord(text[0]) - ord("a"), int(text[1]), ord(text[2]) - ord("a"), int(text[3])


@dataclass(frozen=True)
class BoardCalibration:
    """棋盘四角在窗口截图中的归一化坐标。

    四角依次是左上、右上、右下、左下，坐标均除以截图宽高，因此微信窗口
    平移或按比例缩放后仍可复用。四边形内部使用双线性插值，允许棋盘轻微倾斜。

    参数:
        corners: 四个归一化角点，顺序为左上、右上、右下、左下。
        red_at_bottom: ``True`` 表示红方位于屏幕下方；``False`` 表示棋盘旋转 180°。
    """

    corners: tuple[Point, Point, Point, Point]
    red_at_bottom: bool = True

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("board calibration requires exactly four corners")
        for point in self.corners:
            if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
                raise ValueError("normalized board corners must be inside [0, 1]")

    @classmethod
    def from_pixel_corners(
        cls,
        corners: Iterable[Point],
        image_width: int,
        image_height: int,
        red_at_bottom: bool = True,
    ) -> "BoardCalibration":
        """把截图像素角点转换成可持久化的归一化标定。

        参数:
            corners: 左上、右上、右下、左下四个棋盘交叉点的截图像素坐标。
            image_width: 标定截图宽度。
            image_height: 标定截图高度。
            red_at_bottom: 标定时红方是否位于屏幕下方。
        """

        if image_width <= 0 or image_height <= 0:
            raise ValueError("image width and height must be positive")
        points = tuple(corners)
        if len(points) != 4:
            raise ValueError("exactly four pixel corners are required")
        normalized = tuple(
            Point(point.x / image_width, point.y / image_height) for point in points
        )
        return cls(normalized, red_at_bottom)  # type: ignore[arg-type]

    def grid_point_normalized(self, file_index: int, rank: int) -> Point:
        """取得一个引擎棋盘点在窗口内的归一化坐标。

        参数:
            file_index: 引擎列号，a～i 对应 0～8。
            rank: 引擎行号 0～9，其中 0 为红方底线。
        """

        if not 0 <= file_index < 9 or not 0 <= rank < 10:
            raise ValueError("board coordinate is outside the 9x10 grid")

        if self.red_at_bottom:
            horizontal = file_index / 8.0
            vertical = (9 - rank) / 9.0
        else:
            horizontal = (8 - file_index) / 8.0
            vertical = rank / 9.0

        top_left, top_right, bottom_right, bottom_left = self.corners
        top = Point(
            top_left.x + (top_right.x - top_left.x) * horizontal,
            top_left.y + (top_right.y - top_left.y) * horizontal,
        )
        bottom = Point(
            bottom_left.x + (bottom_right.x - bottom_left.x) * horizontal,
            bottom_left.y + (bottom_right.y - bottom_left.y) * horizontal,
        )
        return Point(
            top.x + (bottom.x - top.x) * vertical,
            top.y + (bottom.y - top.y) * vertical,
        )

    def grid_point_screen(self, file_index: int, rank: int, window: Rect) -> Point:
        """把引擎棋盘点映射到绝对屏幕坐标。

        参数:
            file_index: 引擎列号，a～i 对应 0～8。
            rank: 引擎行号 0～9，其中 0 为红方底线。
            window: 当前微信小程序窗口的屏幕区域。
        """

        point = self.grid_point_normalized(file_index, rank)
        return Point(
            window.left + point.x * window.width,
            window.top + point.y * window.height,
        )

    def move_screen_points(self, move: str, window: Rect) -> tuple[Point, Point]:
        """将一个引擎走法换算为起点和终点的屏幕坐标。

        参数:
            move: 形如 ``b0c2`` 的引擎坐标走法。
            window: 当前微信小程序窗口的屏幕区域。
        """

        from_file, from_rank, to_file, to_rank = parse_move(move)
        return (
            self.grid_point_screen(from_file, from_rank, window),
            self.grid_point_screen(to_file, to_rank, window),
        )
