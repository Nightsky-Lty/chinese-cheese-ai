"""C++ 引擎输出解析的轻量测试。"""

from __future__ import annotations

import unittest

from gui.engine_client import _SEARCH_PATTERN


class EngineClientTests(unittest.TestCase):
    def test_parse_final_search_line(self) -> None:
        line = (
            "search: alpha-beta/pvs depth 4 score 23 nodes 100 qnodes 50 "
            "cutoffs 12 tthits 3 ttcutoffs 1 ttmoves 2 pvsresearches 0 bestmove b0c2"
        )
        match = _SEARCH_PATTERN.match(line)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group("move"), "b0c2")
        self.assertEqual(int(match.group("score")), 23)


if __name__ == "__main__":
    unittest.main()
