"""基于棋盘标定和本地模板的 JJ 象棋棋子识别。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import BoardCalibration, Point


_INITIAL_ROWS = {
    9: "rnbakabnr",
    7: ".c.....c.",
    6: "p.p.p.p.p",
    3: "P.P.P.P.P",
    2: ".C.....C.",
    0: "RNBAKABNR",
}
_PIECE_LABELS = "KABNRCPkabnrcp"
_TEMPLATE_SIZE = 64
_SEARCH_SIZE = 96


@dataclass(frozen=True)
class PieceObservation:
    """一个棋盘交叉点的识别结果。

    参数:
        file_index: 引擎列号，a～i 对应 0～8。
        rank: 引擎行号，0 为红方底线。
        piece: FEN 棋子字符；空位为 ``.``，低置信度棋子为 ``?``。
        confidence: 最佳模板相关系数；空位固定为 1。
        occupancy: 棋子金色外圈的检测比例。
    """

    file_index: int
    rank: int
    piece: str
    confidence: float
    occupancy: float


@dataclass(frozen=True)
class RecognitionResult:
    """完整 9×10 棋盘的识别结果。"""

    observations: tuple[PieceObservation, ...]

    def piece_at(self, file_index: int, rank: int) -> str:
        """返回指定引擎坐标的识别字符。"""

        for observation in self.observations:
            if observation.file_index == file_index and observation.rank == rank:
                return observation.piece
        raise ValueError("board coordinate is missing from recognition result")

    @property
    def unknown_count(self) -> int:
        """返回低置信度、无法确定类型的棋子数量。"""

        return sum(observation.piece == "?" for observation in self.observations)

    def to_fen(self, side_to_move: str) -> str:
        """转换为 C++ 引擎可读取的中国象棋 FEN。

        参数:
            side_to_move: ``red``/``r``/``w`` 或 ``black``/``b``。

        异常:
            ValueError: 阵营无效，或棋盘仍包含未知棋子。
        """

        normalized_side = side_to_move.strip().lower()
        if normalized_side in {"red", "r", "w"}:
            fen_side = "w"
        elif normalized_side in {"black", "b"}:
            fen_side = "b"
        else:
            raise ValueError("side to move must be red or black")
        if self.unknown_count:
            raise ValueError(
                f"cannot create FEN while {self.unknown_count} pieces are unknown"
            )

        rows: list[str] = []
        for rank in range(9, -1, -1):
            empty_count = 0
            encoded = ""
            for file_index in range(9):
                piece = self.piece_at(file_index, rank)
                if piece == ".":
                    empty_count += 1
                    continue
                if empty_count:
                    encoded += str(empty_count)
                    empty_count = 0
                encoded += piece
            if empty_count:
                encoded += str(empty_count)
            rows.append(encoded)
        return "/".join(rows) + f" {fen_side}"


@dataclass(frozen=True)
class TemplateLibrary:
    """按 FEN 棋子字符保存的标准化灰度模板集合。"""

    templates: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        missing = [label for label in _PIECE_LABELS if label not in self.templates]
        if missing:
            raise ValueError(f"template library is missing pieces: {''.join(missing)}")
        for label, images in self.templates.items():
            if label not in _PIECE_LABELS:
                raise ValueError(f"unsupported template label: {label}")
            if images.ndim != 3 or images.shape[1:] != (_TEMPLATE_SIZE, _TEMPLATE_SIZE):
                raise ValueError(f"invalid template shape for {label}: {images.shape}")

    def save(self, path: str | Path) -> Path:
        """将模板保存为不使用 pickle 的压缩 NumPy 文件。"""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "version": np.asarray([1], dtype=np.int32),
        }
        payload.update({f"piece_{label}": images for label, images in self.templates.items()})
        np.savez_compressed(destination, **payload)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary":
        """读取由 :meth:`save` 写出的模板文件。"""

        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            if "version" not in archive or int(archive["version"][0]) != 1:
                raise ValueError("unsupported piece template version")
            templates = {
                label: np.asarray(archive[f"piece_{label}"], dtype=np.uint8)
                for label in _PIECE_LABELS
            }
        return cls(templates)


def _grid_pixel(
    calibration: BoardCalibration,
    image: np.ndarray,
    file_index: int,
    rank: int,
) -> Point:
    """将引擎棋盘坐标换算为截图像素坐标。"""

    height, width = image.shape[:2]
    normalized = calibration.grid_point_normalized(file_index, rank)
    return Point(normalized.x * width, normalized.y * height)


def _grid_spacing(calibration: BoardCalibration, image: np.ndarray) -> float:
    """估算截图中相邻棋盘交叉点的像素距离。"""

    distances: list[float] = []
    for rank in (0, 4, 9):
        for file_index in range(8):
            first = _grid_pixel(calibration, image, file_index, rank)
            second = _grid_pixel(calibration, image, file_index + 1, rank)
            distances.append(float(np.hypot(second.x - first.x, second.y - first.y)))
    for file_index in (0, 4, 8):
        for rank in range(9):
            first = _grid_pixel(calibration, image, file_index, rank)
            second = _grid_pixel(calibration, image, file_index, rank + 1)
            distances.append(float(np.hypot(second.x - first.x, second.y - first.y)))
    spacing = float(np.median(distances))
    if spacing < 20:
        raise ValueError("calibrated board is too small for piece recognition")
    return spacing


def _crop_square(
    image: np.ndarray,
    center: Point,
    half_size: int,
) -> np.ndarray:
    """围绕中心裁剪正方形，靠近图像边界时使用反射填充。"""

    center_x = round(center.x)
    center_y = round(center.y)
    height, width = image.shape[:2]
    left = center_x - half_size
    right = center_x + half_size + 1
    top = center_y - half_size
    bottom = center_y + half_size + 1
    if left >= 0 and top >= 0 and right <= width and bottom <= height:
        return image[top:bottom, left:right]

    padding = half_size + 2
    padded = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_REFLECT_101
    )
    x = center_x + padding
    y = center_y + padding
    return padded[
        y - half_size : y + half_size + 1,
        x - half_size : x + half_size + 1,
    ]


def build_initial_templates(
    image: np.ndarray,
    calibration: BoardCalibration,
) -> TemplateLibrary:
    """从 JJ 象棋标准初始局面截图生成十四类棋子模板。

    参数:
        image: OpenCV BGR 截图，必须是未走子标准初始局面。
        calibration: 与截图对应的棋盘四角和朝向。
    """

    spacing = _grid_spacing(calibration, image)
    half_size = max(16, round(spacing * 0.41))
    collected: dict[str, list[np.ndarray]] = {label: [] for label in _PIECE_LABELS}
    for rank, row in _INITIAL_ROWS.items():
        for file_index, label in enumerate(row):
            if label == ".":
                continue
            center = _grid_pixel(calibration, image, file_index, rank)
            patch = _crop_square(image, center, half_size)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            normalized = cv2.resize(
                gray, (_TEMPLATE_SIZE, _TEMPLATE_SIZE), interpolation=cv2.INTER_AREA
            )
            collected[label].append(normalized)
    return TemplateLibrary(
        {
            label: np.stack(images).astype(np.uint8)
            for label, images in collected.items()
        }
    )


class TemplatePieceRecognizer:
    """使用颜色占位检测和模板相关性识别棋子的轻量识别器。"""

    def __init__(
        self,
        library: TemplateLibrary,
        *,
        occupancy_threshold: float = 0.28,
        confidence_threshold: float = 0.30,
    ) -> None:
        """创建模板识别器。

        参数:
            library: 从同一 JJ 象棋主题初始局面生成的模板。
            occupancy_threshold: 金色棋子外圈像素比例阈值。
            confidence_threshold: 最佳模板相关系数的最低接受值。
        """

        if not 0 < occupancy_threshold < 1:
            raise ValueError("occupancy threshold must be between zero and one")
        if not -1 < confidence_threshold < 1:
            raise ValueError("confidence threshold must be between -1 and one")
        self.library = library
        self.occupancy_threshold = occupancy_threshold
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _occupancy_and_side(patch: np.ndarray) -> tuple[float, bool]:
        """返回棋子占位比例，以及检测到的棋子是否为红方。"""

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        height, width = patch.shape[:2]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        yy, xx = np.ogrid[:height, :width]
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        radius = min(width, height) / 2.0
        ring = (distance > radius * 0.68) & (distance < radius * 0.98)
        core = distance < radius * 0.65
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        # JJ 棋子的外沿是金色；绿色选中圈和白色落子提示虽也会提高
        # 饱和度，却不应被当成棋子。OpenCV 的 HSV 色相范围为 0～179。
        golden_ring = (hue >= 8) & (hue <= 34) & (saturation > 60)
        occupancy = float(np.mean(golden_ring[ring]))
        red_piece = float(np.mean(saturation[core])) > 90.0
        return occupancy, red_piece

    def _classify(self, search_patch: np.ndarray, red_piece: bool) -> tuple[str, float]:
        """在指定阵营的七类模板中寻找最佳匹配。"""

        gray = cv2.cvtColor(search_patch, cv2.COLOR_BGR2GRAY)
        search = cv2.resize(gray, (_SEARCH_SIZE, _SEARCH_SIZE), interpolation=cv2.INTER_AREA)
        labels = "KABNRCP" if red_piece else "kabnrcp"
        best_label = "?"
        best_score = -1.0
        for label in labels:
            for template in self.library.templates[label]:
                scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                score = float(np.nanmax(scores))
                if score > best_score:
                    best_score = score
                    best_label = label
        if best_score < self.confidence_threshold:
            return "?", best_score
        return best_label, best_score

    def _locate_piece_center(
        self,
        image: np.ndarray,
        grid_center: Point,
        occupancy_half_size: int,
        spacing: float,
    ) -> tuple[Point, float, bool]:
        """在交叉点附近寻找金色圆环响应最高的实际棋子中心。

        参数:
            image: 当前窗口截图。
            grid_center: 标定得到的固定棋盘交叉点。
            occupancy_half_size: 占位检测图块的半边长。
            spacing: 相邻棋盘交叉点的像素距离。

        JJ 的选中棋子会略微浮起，因此固定中心检测可能把它误判为空位。搜索范围
        限制在不足五分之一格内，不会越过到相邻棋盘交叉点。
        """

        centered_patch = _crop_square(image, grid_center, occupancy_half_size)
        centered_occupancy, centered_red_piece = self._occupancy_and_side(
            centered_patch
        )
        if centered_occupancy >= self.occupancy_threshold:
            return grid_center, centered_occupancy, centered_red_piece

        # 完全空白格不进行邻域搜索，避免棋盘边缘装饰被误认为浮起棋子。真正略微
        # 偏移的棋子仍会在固定圆环内留下部分金色像素。
        relocation_floor = self.occupancy_threshold * 0.25
        if centered_occupancy < relocation_floor:
            return grid_center, centered_occupancy, centered_red_piece

        max_offset = max(2, round(spacing * 0.16))
        half_offset = max(1, round(max_offset / 2))
        horizontal_offsets = (-max_offset, 0, max_offset)
        vertical_offsets = (
            -max_offset,
            -half_offset,
            0,
            half_offset,
            max_offset,
        )
        best_center = grid_center
        best_occupancy = centered_occupancy
        best_red_piece = centered_red_piece
        for vertical in vertical_offsets:
            for horizontal in horizontal_offsets:
                candidate = Point(
                    grid_center.x + horizontal,
                    grid_center.y + vertical,
                )
                patch = _crop_square(image, candidate, occupancy_half_size)
                occupancy, red_piece = self._occupancy_and_side(patch)
                if occupancy > best_occupancy:
                    best_center = candidate
                    best_occupancy = occupancy
                    best_red_piece = red_piece
        return best_center, best_occupancy, best_red_piece

    def recognize(
        self,
        image: np.ndarray,
        calibration: BoardCalibration,
    ) -> RecognitionResult:
        """识别截图中的全部 90 个棋盘交叉点。"""

        spacing = _grid_spacing(calibration, image)
        occupancy_half_size = max(18, round(spacing * 0.46))
        search_half_size = max(24, round(spacing * 0.62))
        observations: list[PieceObservation] = []
        for rank in range(10):
            for file_index in range(9):
                grid_center = _grid_pixel(calibration, image, file_index, rank)
                center, occupancy, red_piece = self._locate_piece_center(
                    image,
                    grid_center,
                    occupancy_half_size,
                    spacing,
                )
                if occupancy < self.occupancy_threshold:
                    observations.append(
                        PieceObservation(file_index, rank, ".", 1.0, occupancy)
                    )
                    continue
                search_patch = _crop_square(image, center, search_half_size)
                piece, confidence = self._classify(search_patch, red_piece)
                observations.append(
                    PieceObservation(file_index, rank, piece, confidence, occupancy)
                )
        return RecognitionResult(tuple(observations))


def draw_recognition_overlay(
    image: np.ndarray,
    calibration: BoardCalibration,
    result: RecognitionResult,
) -> np.ndarray:
    """在截图上标记识别到的棋子字符和置信度。"""

    overlay = image.copy()
    spacing = _grid_spacing(calibration, image)
    radius = max(8, round(spacing * 0.18))
    font_scale = max(0.35, spacing / 150.0)
    for observation in result.observations:
        if observation.piece == ".":
            continue
        center = _grid_pixel(
            calibration, image, observation.file_index, observation.rank
        )
        point = (round(center.x), round(center.y))
        color = (0, 200, 0) if observation.piece != "?" else (0, 0, 255)
        cv2.circle(overlay, point, radius, color, 2)
        label = f"{observation.piece}:{observation.confidence:.2f}"
        cv2.putText(
            overlay,
            label,
            (point[0] - radius, point[1] - radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
