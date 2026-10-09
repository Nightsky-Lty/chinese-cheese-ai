"""棋盘识别结果编码和模板持久化测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from gui.geometry import Point
from gui.recognition import (
    PieceObservation,
    RecognitionResult,
    TemplateLibrary,
    TemplatePieceRecognizer,
)


def initial_result() -> RecognitionResult:
    """构造标准初始局面的确定性识别结果。"""

    rows = {
        9: "rnbakabnr",
        7: ".c.....c.",
        6: "p.p.p.p.p",
        3: "P.P.P.P.P",
        2: ".C.....C.",
        0: "RNBAKABNR",
    }
    observations = []
    for rank in range(10):
        row = rows.get(rank, ".........")
        for file_index, piece in enumerate(row):
            observations.append(PieceObservation(file_index, rank, piece, 1.0, 1.0))
    return RecognitionResult(tuple(observations))


class RecognitionTests(unittest.TestCase):
    def test_initial_position_fen(self) -> None:
        self.assertEqual(
            initial_result().to_fen("red"),
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
        )

    def test_unknown_piece_blocks_fen(self) -> None:
        observations = list(initial_result().observations)
        observations[0] = PieceObservation(0, 0, "?", 0.1, 0.9)
        with self.assertRaises(ValueError):
            RecognitionResult(tuple(observations)).to_fen("black")

    def test_template_round_trip_without_pickle(self) -> None:
        templates = {
            label: np.full((1, 64, 64), index, dtype=np.uint8)
            for index, label in enumerate("KABNRCPkabnrcp")
        }
        library = TemplateLibrary(templates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pieces.npz"
            library.save(path)
            loaded = TemplateLibrary.load(path)
        for label in templates:
            np.testing.assert_array_equal(loaded.templates[label], templates[label])

    def test_occupancy_center_follows_a_slightly_lifted_piece(self) -> None:
        templates = {
            label: np.zeros((1, 64, 64), dtype=np.uint8)
            for label in "KABNRCPkabnrcp"
        }
        recognizer = TemplatePieceRecognizer(TemplateLibrary(templates))
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        grid_center = Point(100, 100)
        cv2.circle(image, (100, 90), 27, (0, 180, 220), 5)

        center, occupancy, _red_piece = recognizer._locate_piece_center(
            image, grid_center, occupancy_half_size=32, spacing=70
        )

        self.assertLess(center.y, grid_center.y)
        self.assertGreater(occupancy, recognizer.occupancy_threshold)

    def test_green_highlight_ring_is_not_a_piece(self) -> None:
        """高饱和的绿色选中圈不能触发金色棋子占位检测。"""

        patch = np.zeros((65, 65, 3), dtype=np.uint8)
        cv2.circle(patch, (32, 32), 27, (30, 255, 30), 7)

        occupancy, _red_piece = TemplatePieceRecognizer._occupancy_and_side(patch)

        self.assertLess(occupancy, 0.28)


if __name__ == "__main__":
    unittest.main()
