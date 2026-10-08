"""稳定局面与走法差分测试。"""

from __future__ import annotations

import unittest

from gui.board_state import BoardState, StableBoardDetector, detect_move


INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class BoardStateTests(unittest.TestCase):
    def test_detect_normal_move(self) -> None:
        before = BoardState.from_fen(INITIAL_FEN)
        after = BoardState.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b"
        )
        detected = detect_move(before, after, "red", {"b0c2", "h0g2"})
        self.assertEqual(detected.move, "b0c2")
        self.assertEqual(detected.piece, "N")
        self.assertEqual(detected.captured, ".")

    def test_detect_capture(self) -> None:
        before = BoardState.from_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w")
        after = BoardState.from_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 b")
        detected = detect_move(before, after, "red", {"e4e5"})
        self.assertEqual(detected.move, "e4e5")
        self.assertEqual(detected.captured, "p")

    def test_illegal_or_noisy_difference_is_rejected(self) -> None:
        before = BoardState.from_fen(INITIAL_FEN)
        after = BoardState.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b"
        )
        with self.assertRaises(ValueError):
            detect_move(before, after, "red", {"h0g2"})

    def test_stability_requires_consecutive_frames(self) -> None:
        first = BoardState.from_fen(INITIAL_FEN)
        second = BoardState.from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b"
        )
        detector = StableBoardDetector(required_frames=3)
        self.assertIsNone(detector.update(first))
        self.assertIsNone(detector.update(first))
        self.assertEqual(detector.update(first), first)
        self.assertIsNone(detector.update(first))
        self.assertIsNone(detector.update(second))
        self.assertIsNone(detector.update(first))
        self.assertIsNone(detector.update(second))
        self.assertIsNone(detector.update(second))
        self.assertEqual(detector.update(second), second)


if __name__ == "__main__":
    unittest.main()
