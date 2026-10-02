"""实现与 C++ 引擎完全一致的 HalfKA 稀疏特征编码。

棋盘采用引擎内部方向：FEN 第一行是黑方底线，位置编号为
``square = row * 9 + column``。黑方视角会把棋盘旋转 180 度，使观察方始终
位于棋盘下方。修改本文件的编号规则时，必须同步修改 C++ ``halfka.cpp``。
"""

from __future__ import annotations

from dataclasses import dataclass

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
KING_BUCKETS = 9
PIECE_CHANNELS = 14
FEATURE_DIMENSIONS = KING_BUCKETS * PIECE_CHANNELS * BOARD_SIZE

RED = 0
BLACK = 1

# 零基棋子类型与 C++ 的 PieceType::King ... PieceType::Pawn 减一后一致。
# FEN 中 b/e 都表示象，n/h 都表示马，因此映射到相同通道。
PIECE_TYPES = {
    "k": 0,
    "a": 1,
    "b": 2,
    "e": 2,
    "n": 3,
    "h": 3,
    "r": 4,
    "c": 5,
    "p": 6,
}


@dataclass(frozen=True)
class PieceOnBoard:
    """FEN 解析得到的一个非空棋子。

    属性:
        color: ``RED`` 或 ``BLACK``。
        piece_type: 零基棋子类型，范围 0～6。
        square: 引擎内部棋盘位置，范围 0～89。
    """

    color: int
    piece_type: int
    square: int


@dataclass(frozen=True)
class EncodedPosition:
    """一个局面的双视角稀疏特征和行棋方。

    属性:
        red_features: 红方视角下激活的已排序特征编号。
        black_features: 黑方视角下激活的已排序特征编号。
        red_to_move: 当前是否轮到红方行棋。
    """

    red_features: tuple[int, ...]
    black_features: tuple[int, ...]
    red_to_move: bool


def orient_square(square: int, perspective: int) -> int:
    """把位置转换为观察方始终位于下方的标准方向。

    参数:
        square: 原始棋盘位置，范围 0～89。
        perspective: ``RED`` 或 ``BLACK``；黑方视角旋转 180 度。

    返回:
        标准方向下的位置编号。

    异常:
        ValueError: 位置越界或视角不是合法阵营。
    """

    if not 0 <= square < BOARD_SIZE:
        raise ValueError(f"square outside board: {square}")
    if perspective not in (RED, BLACK):
        raise ValueError(f"invalid perspective: {perspective}")
    return square if perspective == RED else BOARD_SIZE - 1 - square


def _king_bucket(king_square: int, perspective: int) -> int:
    """把观察方将帅位置转换为 0～8 的九宫桶编号。

    参数:
        king_square: 观察方将帅的原始棋盘位置。
        perspective: 将帅所属的观察方。

    返回:
        按标准方向从九宫上方到下方排列的桶编号。

    异常:
        ValueError: 将帅不在观察方九宫内。
    """

    oriented = orient_square(king_square, perspective)
    row, col = divmod(oriented, BOARD_COLS)
    if not (7 <= row <= 9 and 3 <= col <= 5):
        raise ValueError("perspective king must be inside its palace")
    return (row - 7) * 3 + (col - 3)


def feature_index(
    perspective_king_square: int,
    perspective: int,
    piece: PieceOnBoard,
) -> int:
    """计算一个棋子对应的 C++ 兼容 HalfKA 特征编号。

    参数:
        perspective_king_square: 观察方将帅的原始位置。
        perspective: 当前编码采用的红方或黑方视角。
        piece: 要编码的棋子及其位置。

    返回:
        范围为 ``[0, FEATURE_DIMENSIONS)`` 的特征编号。
    """

    bucket = _king_bucket(perspective_king_square, perspective)
    # 前 7 个通道表示己方七种棋子，后 7 个通道表示对方七种棋子。
    relative_color = 0 if piece.color == perspective else 1
    channel = relative_color * 7 + piece.piece_type
    oriented_piece_square = orient_square(piece.square, perspective)
    return (bucket * PIECE_CHANNELS + channel) * BOARD_SIZE + oriented_piece_square


def _parse_fen(fen: str) -> tuple[list[PieceOnBoard], int, dict[int, int]]:
    """解析训练所需的棋子、行棋方和将帅位置。

    参数:
        fen: 至少包含棋子布局和行棋方的中国象棋 FEN。

    返回:
        ``(棋子列表, 行棋方, 双方将帅位置)``。

    异常:
        ValueError: FEN 结构、棋子字符、行宽或将帅数量不合法。
    """

    fields = fen.split()
    if len(fields) < 2:
        raise ValueError("FEN must contain placement and side-to-move")

    ranks = fields[0].split("/")
    if len(ranks) != BOARD_ROWS:
        raise ValueError("FEN must contain exactly ten ranks")

    pieces: list[PieceOnBoard] = []
    kings: dict[int, int] = {}
    king_counts = {RED: 0, BLACK: 0}
    for row, rank in enumerate(ranks):
        col = 0
        for token in rank:
            if token.isdigit():
                empty_count = int(token)
                if empty_count <= 0:
                    raise ValueError("FEN empty-square count must be positive")
                col += empty_count
                continue

            key = token.lower()
            if key not in PIECE_TYPES:
                raise ValueError(f"unknown FEN piece: {token}")
            if col >= BOARD_COLS:
                raise ValueError("too many squares in FEN rank")

            color = RED if token.isupper() else BLACK
            piece_type = PIECE_TYPES[key]
            square = row * BOARD_COLS + col
            pieces.append(PieceOnBoard(color, piece_type, square))
            if piece_type == 0:
                king_counts[color] += 1
                kings[color] = square
            col += 1

        if col != BOARD_COLS:
            raise ValueError("each FEN rank must contain exactly nine squares")

    if king_counts != {RED: 1, BLACK: 1}:
        raise ValueError("FEN must contain exactly one king per side")

    side = fields[1].lower()
    if side in ("w", "r"):
        side_to_move = RED
    elif side == "b":
        side_to_move = BLACK
    else:
        raise ValueError(f"invalid FEN side-to-move: {fields[1]}")
    return pieces, side_to_move, kings


def encode_fen(fen: str) -> EncodedPosition:
    """把中国象棋 FEN 编码为红黑双方的 HalfKA 稀疏输入。

    参数:
        fen: 至少包含棋子布局和行棋方的中国象棋 FEN。

    返回:
        包含两个视角特征和行棋方的 ``EncodedPosition``。

    异常:
        ValueError: FEN 无法形成有效的 HalfKA 输入。
    """

    pieces, side_to_move, kings = _parse_fen(fen)
    red_features = tuple(
        sorted(feature_index(kings[RED], RED, piece) for piece in pieces)
    )
    black_features = tuple(
        sorted(feature_index(kings[BLACK], BLACK, piece) for piece in pieces)
    )
    return EncodedPosition(
        red_features=red_features,
        black_features=black_features,
        red_to_move=side_to_move == RED,
    )
