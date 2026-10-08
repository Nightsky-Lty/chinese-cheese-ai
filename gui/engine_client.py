"""通过标准命令行接口调用现有 C++ 象棋引擎。"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_SEARCH_PATTERN = re.compile(
    r"^search:.*\bdepth (?P<depth>\d+)\s+score (?P<score>-?\d+).*"
    r"\bbestmove (?P<move>[a-i][0-9][a-i][0-9])\s*$"
)


@dataclass(frozen=True)
class EngineAnalysis:
    """C++ 引擎对识别局面的搜索结果。"""

    fen: str
    best_move: str
    score: int
    depth: int
    raw_output: str


def analyze_fen(
    engine: str | Path,
    fen: str,
    *,
    depth: int = 4,
    nnue: str | Path | None = None,
    timeout: float = 30.0,
) -> EngineAnalysis:
    """启动 C++ CLI 搜索 FEN，并解析最终最佳走法。

    参数:
        engine: ``xiangqi_cli`` 可执行文件路径。
        fen: 识别器生成的完整 FEN。
        depth: 固定搜索深度，必须为正数。
        nnue: 可选 NNUE 权重文件；省略时使用手工评估。
        timeout: 子进程最长运行秒数。
    """

    if depth <= 0:
        raise ValueError("engine search depth must be positive")
    if timeout <= 0:
        raise ValueError("engine timeout must be positive")
    executable = Path(engine)
    if not executable.is_file():
        raise RuntimeError(f"engine executable does not exist: {executable}")
    command = [str(executable), "--fen", fen, "--depth", str(depth)]
    if nnue is not None:
        command.extend(["--nnue", str(nnue)])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"engine search exceeded {timeout:g} seconds") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"engine search failed: {detail}")
    for line in reversed(completed.stdout.splitlines()):
        match = _SEARCH_PATTERN.match(line.strip())
        if match:
            return EngineAnalysis(
                fen=fen,
                best_move=match.group("move"),
                score=int(match.group("score")),
                depth=int(match.group("depth")),
                raw_output=completed.stdout,
            )
    raise RuntimeError("engine output did not contain a final best move")
