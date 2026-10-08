"""棋盘坐标换算的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gui.config import GuiConfig, load_config, save_config
from gui.control import execute_move
from gui.geometry import BoardCalibration, Point, Rect, parse_move
from gui.windows import WindowInfo


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = BoardCalibration(
            (
                Point(0.1, 0.2),
                Point(0.9, 0.2),
                Point(0.9, 0.95),
                Point(0.1, 0.95),
            )
        )
        self.window = Rect(100, 50, 1000, 800)

    def test_parse_move(self) -> None:
        self.assertEqual(parse_move("b0c2"), (1, 0, 2, 2))
        with self.assertRaises(ValueError):
            parse_move("j0a0")

    def test_red_bottom_mapping(self) -> None:
        top_left = self.calibration.grid_point_screen(0, 9, self.window)
        bottom_right = self.calibration.grid_point_screen(8, 0, self.window)
        self.assertAlmostEqual(top_left.x, 200)
        self.assertAlmostEqual(top_left.y, 210)
        self.assertAlmostEqual(bottom_right.x, 1000)
        self.assertAlmostEqual(bottom_right.y, 810)

    def test_black_bottom_rotates_board(self) -> None:
        calibration = BoardCalibration(self.calibration.corners, red_at_bottom=False)
        a_zero = calibration.grid_point_screen(0, 0, self.window)
        self.assertAlmostEqual(a_zero.x, 1000)
        self.assertAlmostEqual(a_zero.y, 210)

    def test_trapezoid_uses_bilinear_interpolation(self) -> None:
        calibration = BoardCalibration(
            (Point(0.2, 0.1), Point(0.8, 0.1), Point(0.9, 0.9), Point(0.1, 0.9))
        )
        center = calibration.grid_point_normalized(4, 4)
        self.assertAlmostEqual(center.x, 0.5)
        self.assertGreater(center.y, 0.5)

    def test_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.json"
            save_config(GuiConfig("微信", self.calibration), path)
            loaded = load_config(path)
            self.assertEqual(loaded, GuiConfig("微信", self.calibration))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], 1)

    def test_move_is_dry_run_by_default(self) -> None:
        window = WindowInfo(1, "微信", "象棋", self.window, 2)
        result = execute_move(window, self.calibration, "a0i9")
        self.assertFalse(result.executed)
        self.assertEqual(result.source, Point(200, 810))
        self.assertEqual(result.target, Point(1000, 210))


if __name__ == "__main__":
    unittest.main()
