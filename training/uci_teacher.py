"""通过标准 UCI 调用离线教师，严格区分精确评分、边界和将杀。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

from .halfka import encode_fen


@dataclass(frozen=True)
class TeacherAnalysis:
    """教师结果；cp 是当前行棋方视角，mate 不转换成普通评分。"""

    score_kind: str
    score: int
    depth: int
    nodes: int
    elapsed_ms: int
    best_move: str
    pv: tuple[str, ...]
    bound: str | None


def full_uci_fen(fen: str) -> str:
    """将项目的两字段 FEN 补齐为教师所需的六字段，不伪造历史走法。"""

    encode_fen(fen)
    fields = fen.split()
    if fields[1] == "r":
        fields[1] = "w"
    if len(fields) == 2:
        fields.extend(["-", "-", "0", "1"])
    if len(fields) != 6 or any(char in fen for char in "\r\n"):
        raise ValueError("teacher FEN must have two or six fields and no newlines")
    return " ".join(fields)


def parse_analysis(line: str) -> TeacherAnalysis | None:
    """解析一条主变信息；其他 MultiPV、日志和不完整评分不会成为标签。"""

    if not line.startswith("info "):
        return None
    tokens = line.split()
    if "multipv" in tokens and tokens[tokens.index("multipv") + 1] != "1":
        return None
    if not all(key in tokens for key in ("score", "depth", "pv")):
        return None
    try:
        kind_index = tokens.index("score") + 1
        kind, score = tokens[kind_index], int(tokens[kind_index + 1])
        if kind not in ("cp", "mate"):
            return None
        pv = tuple(tokens[tokens.index("pv") + 1:])
        if not pv:
            return None
        return TeacherAnalysis(
            kind, score, int(tokens[tokens.index("depth") + 1]),
            int(tokens[tokens.index("nodes") + 1]) if "nodes" in tokens else 0,
            int(tokens[tokens.index("time") + 1]) if "time" in tokens else 0,
            pv[0], pv,
            "lower" if "lowerbound" in tokens else "upper" if "upperbound" in tokens else None,
        )
    except (ValueError, IndexError):
        raise RuntimeError(f"malformed teacher score: {line}") from None


class UciTeacher:
    """单进程 UCI 教师；每局面清空 Hash，避免断点恢复改变标签上下文。"""

    def __init__(self, engine: Path, network: Path, *, threads: int = 1,
                 hash_mb: int = 32, timeout: float = 30.0) -> None:
        """engine/network 是本地文件；timeout 限制握手和单次分析等待。"""

        self.timeout = timeout
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.identity: list[str] = []
        self.options: set[str] = set()
        self.process = subprocess.Popen(
            [str(engine.resolve())], cwd=engine.resolve().parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        try:
            self.send("uci")
            for line in self.until("uciok"):
                if line.startswith("id "):
                    self.identity.append(line)
                if line.startswith("option name "):
                    self.options.add(line.removeprefix("option name ").split(" type ", 1)[0])
            for name, value in (("Threads", threads), ("Hash", hash_mb),
                                ("EvalFile", network.resolve()), ("MultiPV", 1)):
                self.set_option(name, str(value), required=True)
            if "UCI_ShowWDL" in self.options:
                self.set_option("UCI_ShowWDL", "true")
            self.ready()
        except BaseException:
            self.close()
            raise

    def _read_output(self) -> None:
        """后台读管道，主线程用队列截止时间避免永久卡在 readline。"""

        try:
            for line in self.process.stdout:
                self.lines.put(line.rstrip())
        finally:
            self.lines.put(None)

    def send(self, command: str) -> None:
        """发送一条已构造的 UCI 命令，不经 shell，不接受换行注入。"""

        if "\n" in command or "\r" in command:
            raise ValueError("UCI command contains a newline")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def until(self, token: str) -> list[str]:
        """收集到指定响应；引擎退出、严重错误或超时均停止本次标注。"""

        deadline = time.monotonic() + self.timeout
        output: list[str] = []
        while True:
            try:
                line = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"teacher timed out waiting for {token}") from None
            if line is None:
                raise RuntimeError(f"teacher exited waiting for {token}: {output[-8:]}")
            output.append(line)
            if "CRITICAL ERROR" in line or "ERROR" in line:
                raise RuntimeError(f"teacher error: {line}")
            if line == token or line.startswith(token + " "):
                return output
            if time.monotonic() >= deadline:
                raise TimeoutError(f"teacher timed out waiting for {token}")

    def set_option(self, name: str, value: str, required: bool = False) -> None:
        """设置教师实际支持的选项，关键选项缺失时拒绝静默降级。"""

        if name not in self.options:
            if required:
                raise RuntimeError(f"teacher missing required UCI option: {name}")
            return
        self.send(f"setoption name {name} value {value}")

    def ready(self) -> None:
        """等待教师完成前面的配置或清理。"""

        self.send("isready")
        self.until("readyok")

    def analyze(self, fen: str, *, nodes: int = 5000, depth: int | None = None) -> TeacherAnalysis | None:
        """以固定节点或固定深度分析 FEN；没有可走着时返回 None。"""

        self.send("ucinewgame")
        self.ready()
        self.send("position fen " + full_uci_fen(fen))
        self.send(f"go depth {depth}" if depth is not None else f"go nodes {nodes}")
        lines = self.until("bestmove")
        best = lines[-1].split()[1]
        if best in ("(none)", "0000"):
            return None
        if not re.fullmatch(r"[a-i][0-9][a-i][0-9]", best):
            raise RuntimeError(f"invalid teacher bestmove: {best}")
        analyses = [item for line in lines if (item := parse_analysis(line)) is not None]
        if not analyses or analyses[-1].best_move != best:
            raise RuntimeError("teacher bestmove does not match its final scored PV")
        return analyses[-1]

    def close(self) -> None:
        """退出教师；故障时终止子进程，避免留下占用 CPU 的后台分析。"""

        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()

    def __enter__(self) -> UciTeacher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
