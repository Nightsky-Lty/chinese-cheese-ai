#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace xiangqi {

constexpr int kBoardRows = 10;
constexpr int kBoardCols = 9;
constexpr int kBoardSize = kBoardRows * kBoardCols;
constexpr int kNoSquare = -1;

enum class Color : std::uint8_t {
    Red = 0,
    Black = 1,
};

/// @brief 获取给定阵营的对手阵营。
/// @param color 当前阵营。
/// @return `Red` 对应 `Black`，`Black` 对应 `Red`。
constexpr Color opposite(Color color) {
    return color == Color::Red ? Color::Black : Color::Red;
}

enum class PieceType : std::uint8_t {
    Empty = 0,
    King = 1,
    Advisor = 2,
    Elephant = 3,
    Horse = 4,
    Rook = 5,
    Cannon = 6,
    Pawn = 7,
};

using Piece = std::uint8_t;
constexpr Piece kEmpty = 0;
constexpr std::uint8_t kBlackFlag = 0x08;

/// @brief 将阵营和棋子类型编码为一个紧凑的棋子值。
/// @param color 棋子所属阵营；当 `type` 为 `Empty` 时该参数被忽略。
/// @param type 棋子类型。
/// @return 编码后的棋子；空类型返回 `kEmpty`。
constexpr Piece make_piece(Color color, PieceType type) {
    if (type == PieceType::Empty) {
        return kEmpty;
    }
    return static_cast<Piece>(static_cast<std::uint8_t>(type) |
                              (color == Color::Black ? kBlackFlag : 0));
}

/// @brief 从棋子编码中提取棋子类型。
/// @param piece 棋子编码，可以是 `kEmpty`。
/// @return 棋子类型；空位置返回 `PieceType::Empty`。
constexpr PieceType piece_type(Piece piece) {
    return static_cast<PieceType>(piece & 0x07);
}

/// @brief 从非空棋子编码中提取所属阵营。
/// @param piece 非空棋子编码。
/// @return 棋子所属的红方或黑方。
/// @note 不应使用该函数查询 `kEmpty`，因为空位置没有阵营。
constexpr Color piece_color(Piece piece) {
    return (piece & kBlackFlag) != 0 ? Color::Black : Color::Red;
}

/// @brief 判断一个棋子编码是否表示空位置。
/// @param piece 待检查的棋子编码。
/// @return 为空返回 `true`，否则返回 `false`。
constexpr bool is_empty(Piece piece) {
    return piece == kEmpty;
}

/// @brief 将内部行列坐标转换为一维棋盘下标。
/// @param row 内部行号，范围为 0～9，0 为黑方底线。
/// @param col 内部列号，范围为 0～8。
/// @return 范围为 0～89 的棋盘下标。
/// @note 调用者应先确保行列坐标位于棋盘内。
constexpr int make_square(int row, int col) {
    return row * kBoardCols + col;
}

/// @brief 从一维棋盘下标中取得内部行号。
/// @param square 棋盘下标，范围为 0～89。
/// @return 内部行号，范围为 0～9。
constexpr int square_row(int square) {
    return square / kBoardCols;
}

/// @brief 从一维棋盘下标中取得内部列号。
/// @param square 棋盘下标，范围为 0～89。
/// @return 内部列号，范围为 0～8。
constexpr int square_col(int square) {
    return square % kBoardCols;
}

/// @brief 判断一组内部行列坐标是否位于棋盘内。
/// @param row 待检查的内部行号。
/// @param col 待检查的内部列号。
/// @return 坐标有效返回 `true`，否则返回 `false`。
constexpr bool inside_board(int row, int col) {
    return row >= 0 && row < kBoardRows && col >= 0 && col < kBoardCols;
}

struct Move {
    std::uint8_t from{};
    std::uint8_t to{};

    friend constexpr bool operator==(Move, Move) = default;
};

/// @brief 将内部走法转换为 UCI 风格的四字符坐标，例如 `b0c2`。
/// @param move 待转换的走法。
/// @return 由起点和终点组成的四字符字符串；红方底线为第 0 行。
std::string move_to_string(Move move);

/// @brief 解析 UCI 风格的四字符走法。
/// @param text 形如 `b0c2` 的走法字符串，列为 `a`～`i`，行为 `0`～`9`。
/// @return 解析成功时返回走法，否则返回 `std::nullopt`。
std::optional<Move> move_from_string(const std::string& text);

}  // namespace xiangqi
