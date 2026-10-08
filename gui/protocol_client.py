"""C++ 常驻象棋协议进程的 Python 客户端。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProtocolError(RuntimeError):
    """引擎拒绝命令或返回无效响应。"""


@dataclass(frozen=True)
class ProtocolState:
    """常驻引擎当前局面和规则历史摘要。"""

    fen: str
    ply: int
    no_capture_plies: int


@dataclass(frozen=True)
class ProtocolSearchResult:
    """常驻引擎返回的固定深度搜索结果。"""

    best_move: str | None
    score: int
    depth: int
    nodes: int
    timed_out: bool


@dataclass(frozen=True)
class ProtocolGameResult:
    """C++ 规则层返回的当前对局结果及原因。"""

    outcome: str
    reason: str

    @property
    def finished(self) -> bool:
        """对局已经产生胜负或和棋时返回 True。"""

        return self.outcome != "ongoing"


class ProtocolEngineClient:
    """管理一个保存 ``GameHistory`` 的 C++ 引擎子进程。"""

    def __init__(
        self,
        executable: str | Path,
        *,
        nnue: str | Path | None = None,
    ) -> None:
        """启动协议进程并完成版本握手。

        参数:
            executable: ``xiangqi_protocol`` 可执行文件路径。
            nnue: 可选的 NNUE 权重路径；省略时使用手工评估。
        """

        path = Path(executable)
        if not path.is_file():
            raise RuntimeError(f"protocol executable does not exist: {path}")
        command = [str(path)]
        if nnue is not None:
            command.extend(["--nnue", str(nnue)])
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        greeting = self._read_response()
        if greeting != "ready xiangqi-protocol 1":
            self.close(force=True)
            raise ProtocolError(f"unexpected protocol greeting: {greeting}")

    def __enter__(self) -> "ProtocolEngineClient":
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def _read_response(self) -> str:
        """读取一行响应，并在进程异常结束时附带错误输出。"""

        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if line:
            return line.rstrip("\r\n")
        error_output = ""
        if self._process.stderr is not None:
            error_output = self._process.stderr.read().strip()
        raise ProtocolError(
            "engine protocol closed unexpectedly"
            + (f": {error_output}" if error_output else "")
        )

    def _request(self, command: str) -> str:
        """发送一条不含换行的命令，并检查通用错误响应。"""

        if self._process.poll() is not None:
            raise ProtocolError("engine protocol is not running")
        if "\n" in command or "\r" in command:
            raise ValueError("protocol command must fit on one line")
        assert self._process.stdin is not None
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if response.startswith("error "):
            raise ProtocolError(response)
        return response

    @staticmethod
    def _parse_state(text: str) -> ProtocolState:
        """解析 ``state fen ... ply N nocapture N`` 响应片段。"""

        marker = "state fen "
        start = text.find(marker)
        if start < 0:
            raise ProtocolError(f"invalid state response: {text}")
        fields = text[start + len(marker) :]
        try:
            fen, counts = fields.rsplit(" ply ", 1)
            ply_text, no_capture_text = counts.split(" nocapture ", 1)
            return ProtocolState(fen, int(ply_text), int(no_capture_text))
        except (ValueError, TypeError) as error:
            raise ProtocolError(f"invalid state response: {text}") from error

    def ping(self) -> None:
        """验证子进程仍能正常响应。"""

        if self._request("ping") != "pong":
            raise ProtocolError("engine protocol returned an invalid pong")

    def set_position(self, fen: str) -> ProtocolState:
        """设置起始 FEN，并清空 C++ 端历史。"""

        return self._parse_state(self._request(f"position fen {fen}"))

    def state(self) -> ProtocolState:
        """读取当前 FEN、半回合数和无吃子计数。"""

        return self._parse_state(self._request("state"))

    def legal_moves(self) -> tuple[str, ...]:
        """返回当前局面的全部合法四字符走法。"""

        response = self._request("legal")
        fields = response.split()
        if len(fields) < 2 or fields[0] != "legal":
            raise ProtocolError(f"invalid legal response: {response}")
        try:
            count = int(fields[1])
        except ValueError as error:
            raise ProtocolError(f"invalid legal response: {response}") from error
        moves = tuple(fields[2:])
        if len(moves) != count:
            raise ProtocolError(f"legal move count mismatch: {response}")
        return moves

    def result(self) -> ProtocolGameResult:
        """查询当前历史的胜负、和棋或继续状态。"""

        response = self._request("result")
        fields = response.split()
        if len(fields) != 4 or fields[0] != "result" or fields[2] != "reason":
            raise ProtocolError(f"invalid game result response: {response}")
        outcome = fields[1]
        reason = fields[3]
        if outcome not in {"ongoing", "red_win", "black_win", "draw"}:
            raise ProtocolError(f"invalid game result outcome: {response}")
        return ProtocolGameResult(outcome, reason)

    def play(self, move: str) -> ProtocolState:
        """执行并记录一步合法走法。"""

        response = self._request(f"move {move}")
        if not response.startswith(f"ok move {move} "):
            raise ProtocolError(f"invalid move response: {response}")
        return self._parse_state(response)

    def undo(self) -> ProtocolState:
        """撤销 C++ 历史中的最后一步。"""

        return self._parse_state(self._request("undo"))

    def search(
        self, depth: int = 4, *, move_time_ms: int | None = None
    ) -> ProtocolSearchResult:
        """搜索当前历史局面，可限制总思考时间，但不执行最佳走法。"""

        if depth <= 0:
            raise ValueError("search depth must be positive")
        if move_time_ms is not None and move_time_ms <= 0:
            raise ValueError("move time must be positive")
        command = f"go depth {depth}"
        if move_time_ms is not None:
            command += f" movetime {move_time_ms}"
        response = self._request(command)
        fields = response.split()
        if (
            len(fields) != 10
            or fields[0] != "bestmove"
            or fields[2] != "score"
            or fields[4] != "depth"
            or fields[6] != "nodes"
            or fields[8] != "timedout"
        ):
            raise ProtocolError(f"invalid search response: {response}")
        try:
            timed_out = int(fields[9])
            if timed_out not in {0, 1}:
                raise ValueError("timedout must be zero or one")
            return ProtocolSearchResult(
                best_move=None if fields[1] == "none" else fields[1],
                score=int(fields[3]),
                depth=int(fields[5]),
                nodes=int(fields[7]),
                timed_out=bool(timed_out),
            )
        except (ValueError, IndexError) as error:
            raise ProtocolError(f"invalid search response: {response}") from error

    def close(self, *, force: bool = False) -> None:
        """正常关闭协议进程；协议失效时可强制终止。"""

        if self._process.poll() is None:
            if not force:
                try:
                    self._request("quit")
                except (BrokenPipeError, ProtocolError):
                    force = True
            if force and self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        for stream in (
            self._process.stdin,
            self._process.stdout,
            self._process.stderr,
        ):
            if stream is not None and not stream.closed:
                stream.close()
