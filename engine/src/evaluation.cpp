#include "xiangqi/evaluation.hpp"

#include <algorithm>
#include <cmath>

namespace xiangqi {
namespace {

/// @brief 计算棋子所在列靠近棋盘中心的程度。
/// @param col 内部列号，范围为 0～8。
/// @return 中线 e 列为 4，最边缘列为 0。
int file_centrality(int col) {
    return 4 - std::abs(col - 4);
}

/// @brief 计算棋子从己方底线向前推进的行数。
/// @param color 棋子所属阵营。
/// @param row 棋子的内部行号。
/// @return 范围为 0～9 的相对推进距离。
int relative_advance(Color color, int row) {
    return color == Color::Red ? 9 - row : row;
}

/// @brief 判断兵/卒是否已经越过楚河汉界。
/// @param color 兵/卒所属阵营。
/// @param row 兵/卒的内部行号。
/// @return 已过河返回 `true`。
bool pawn_crossed_river(Color color, int row) {
    return color == Color::Red ? row <= 4 : row >= 5;
}

/// @brief 计算单个棋子的轻量位置奖励。
/// @param piece 待评估的非空棋子。
/// @param square 棋子所在的棋盘下标。
/// @return 非负位置奖励；双方使用完全镜像的计算方式。
int positional_bonus(Piece piece, int square) {
    const PieceType type = piece_type(piece);
    const Color color = piece_color(piece);
    const int row = square_row(square);
    const int col = square_col(square);
    const int center = file_centrality(col);
    const int advance = relative_advance(color, row);

    switch (type) {
        case PieceType::Pawn:
            // 越接近对方底线价值越高；过河后获得横向机动能力。
            return advance * 8 + (pawn_crossed_river(color, row) ? 30 : 0) + center * 3;
        case PieceType::Horse:
            // 马在中央通常拥有更多可达位置，边马受到轻微惩罚。
            return center * 6 + std::min(advance, 6) * 2;
        case PieceType::Cannon:
            // 炮在中路和较前位置更容易形成牵制与攻击。
            return center * 4 + std::min(advance, 6) * 2;
        case PieceType::Rook:
            // 车的位置奖励保持较小，避免掩盖其主要子力价值。
            return center * 2 + std::min(advance, 6);
        case PieceType::Advisor:
        case PieceType::Elephant:
        case PieceType::King:
        case PieceType::Empty:
            return 0;
    }
    return 0;
}

}  // namespace

int piece_base_value(PieceType type) {
    switch (type) {
        case PieceType::Rook: return 900;
        case PieceType::Cannon: return 450;
        case PieceType::Horse: return 400;
        case PieceType::Advisor: return 200;
        case PieceType::Elephant: return 200;
        case PieceType::Pawn: return 100;
        case PieceType::King:
        case PieceType::Empty:
            return 0;
    }
    return 0;
}

std::string_view HandcraftedEvaluator::name() const noexcept {
    return "handcrafted";
}

EvaluationBreakdown HandcraftedEvaluator::breakdown(const Position& position) const {
    EvaluationBreakdown result;
    for (int square = 0; square < kBoardSize; ++square) {
        const Piece piece = position.piece_at(square);
        if (is_empty(piece)) {
            continue;
        }

        const int sign = piece_color(piece) == Color::Red ? 1 : -1;
        result.material += sign * piece_base_value(piece_type(piece));
        result.positional += sign * positional_bonus(piece, square);
    }
    return result;
}

int HandcraftedEvaluator::evaluate_for(
    const Position& position, Color perspective) const {
    const int red_score = breakdown(position).total();
    return perspective == Color::Red ? red_score : -red_score;
}

const HandcraftedEvaluator& handcrafted_evaluator() {
    static const HandcraftedEvaluator evaluator;
    return evaluator;
}

EvaluationBreakdown evaluate_breakdown(const Position& position) {
    return handcrafted_evaluator().breakdown(position);
}

int evaluate_for(const Position& position, Color perspective) {
    return handcrafted_evaluator().evaluate_for(position, perspective);
}

int evaluate(const Position& position) {
    return handcrafted_evaluator().evaluate(position);
}

}  // namespace xiangqi
