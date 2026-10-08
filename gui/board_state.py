"""识别局面的稳定化、差分和象棋走法推断。"""

from __future__ import annotations

from dataclasses import dataclass

from .recognition import RecognitionResult


def _piece_side(piece: str) -> str | None:
    """返回 FEN 棋子的 red/black 阵营，空位返回 None。"""

    if piece in "KABNRCP":
        return "red"
    if piece in "kabnrcp":
        return "black"
    return None


def _square_name(index: int) -> str:
    """将 rank-major 的 0～89 下标转换为 a0～i9。"""

    return chr(ord("a") + index % 9) + str(index // 9)


@dataclass(frozen=True)
class BoardState:
    """按 rank-major 顺序保存的 90 格纯棋盘状态。"""

    pieces: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.pieces) != 90:
            raise ValueError("board state must contain exactly 90 squares")
        invalid = [piece for piece in self.pieces if piece not in ".?KABNRCPkabnrcp"]
        if invalid:
            raise ValueError(f"unsupported board pieces: {invalid}")

    @classmethod
    def from_recognition(cls, result: RecognitionResult) -> "BoardState":
        """从单帧识别结果建立棋盘状态。"""

        pieces = ["?"] * 90
        for observation in result.observations:
            pieces[observation.rank * 9 + observation.file_index] = observation.piece
        return cls(tuple(pieces))

    @classmethod
    def from_fen(cls, fen: str) -> "BoardState":
        """读取至少包含棋子布局字段的中国象棋 FEN。"""

        placement = fen.strip().split()[0]
        rows = placement.split("/")
        if len(rows) != 10:
            raise ValueError("FEN must contain ten ranks")
        pieces = ["."] * 90
        for fen_row, rank in zip(rows, range(9, -1, -1)):
            file_index = 0
            for token in fen_row:
                if token.isdigit():
                    file_index += int(token)
                elif token in "KABNRCPkabnrcp":
                    if file_index >= 9:
                        raise ValueError("FEN rank is too wide")
                    pieces[rank * 9 + file_index] = token
                    file_index += 1
                else:
                    raise ValueError(f"unsupported FEN token: {token}")
            if file_index != 9:
                raise ValueError("each FEN rank must contain nine files")
        return cls(tuple(pieces))

    @property
    def has_unknown(self) -> bool:
        """返回棋盘中是否仍有低置信度未知棋子。"""

        return "?" in self.pieces

    def piece_at(self, file_index: int, rank: int) -> str:
        """读取指定引擎坐标上的棋子字符。"""

        if not 0 <= file_index < 9 or not 0 <= rank < 10:
            raise ValueError("board coordinate is outside the 9x10 grid")
        return self.pieces[rank * 9 + file_index]

    def to_fen(self, side_to_move: str) -> str:
        """将棋盘与显式指定的行棋方转换为完整 FEN。"""

        if self.has_unknown:
            raise ValueError("cannot create FEN from a board containing unknown pieces")
        side = side_to_move.strip().lower()
        if side == "red":
            fen_side = "w"
        elif side == "black":
            fen_side = "b"
        else:
            raise ValueError("side to move must be red or black")
        rows: list[str] = []
        for rank in range(9, -1, -1):
            empty = 0
            encoded = ""
            for file_index in range(9):
                piece = self.piece_at(file_index, rank)
                if piece == ".":
                    empty += 1
                else:
                    if empty:
                        encoded += str(empty)
                        empty = 0
                    encoded += piece
            if empty:
                encoded += str(empty)
            rows.append(encoded)
        return "/".join(rows) + f" {fen_side}"


@dataclass(frozen=True)
class DetectedMove:
    """通过两个稳定棋盘差分得到的候选走法。"""

    move: str
    piece: str
    captured: str


def detect_move(
    previous: BoardState,
    current: BoardState,
    side_to_move: str,
    legal_moves: tuple[str, ...] | set[str],
) -> DetectedMove:
    """从前后棋盘推断唯一合法走法。

    普通走法和吃子都必须只改变起点、终点两个格子。候选走法还必须出现在 C++
    引擎根据完整规则历史生成的合法着列表中。
    """

    if previous.has_unknown or current.has_unknown:
        raise ValueError("cannot detect a move while either board contains unknown pieces")
    normalized_side = side_to_move.strip().lower()
    if normalized_side not in {"red", "black"}:
        raise ValueError("side to move must be red or black")
    changed = [
        index
        for index, (before, after) in enumerate(zip(previous.pieces, current.pieces))
        if before != after
    ]
    if len(changed) != 2:
        raise ValueError(f"a move must change exactly two squares, observed {len(changed)}")

    candidates: list[DetectedMove] = []
    for source in changed:
        target = changed[1] if source == changed[0] else changed[0]
        moved_piece = previous.pieces[source]
        captured = previous.pieces[target]
        if _piece_side(moved_piece) != normalized_side:
            continue
        if current.pieces[source] != "." or current.pieces[target] != moved_piece:
            continue
        if captured != "." and _piece_side(captured) == normalized_side:
            continue
        move = _square_name(source) + _square_name(target)
        candidates.append(DetectedMove(move, moved_piece, captured))
    if len(candidates) != 1:
        raise ValueError("board difference does not describe one unambiguous move")
    candidate = candidates[0]
    if candidate.move not in set(legal_moves):
        raise ValueError(f"detected move {candidate.move} is not legal")
    return candidate


class StableBoardDetector:
    """只有连续多帧完全相同时才接受一个新的棋盘状态。"""

    def __init__(self, required_frames: int = 3) -> None:
        if required_frames <= 0:
            raise ValueError("required stable frame count must be positive")
        self.required_frames = required_frames
        self._candidate: BoardState | None = None
        self._candidate_frames = 0
        self._accepted: BoardState | None = None

    @property
    def accepted(self) -> BoardState | None:
        """返回最近一次已经稳定接受的棋盘。"""

        return self._accepted

    def reset(self) -> None:
        """清除候选帧和已经接受的棋盘。"""

        self._candidate = None
        self._candidate_frames = 0
        self._accepted = None

    def restore_accepted(self, board: BoardState | None) -> None:
        """在上层合法性校验失败后恢复最近可信棋盘。"""

        self._candidate = None
        self._candidate_frames = 0
        self._accepted = board

    def update(self, board: BoardState) -> BoardState | None:
        """提交一帧；新棋盘首次达到稳定阈值时返回它，否则返回 None。"""

        if board.has_unknown:
            self._candidate = None
            self._candidate_frames = 0
            return None
        if board == self._accepted:
            self._candidate = None
            self._candidate_frames = 0
            return None
        if board == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = board
            self._candidate_frames = 1
        if self._candidate_frames < self.required_frames:
            return None
        self._accepted = board
        self._candidate = None
        self._candidate_frames = 0
        return board
