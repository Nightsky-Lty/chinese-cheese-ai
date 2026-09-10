#include "xiangqi/halfka.hpp"

#include <algorithm>
#include <stdexcept>

namespace xiangqi {
namespace {

/// @brief 把观察方将帅的标准方向位置转换为九宫桶编号。
/// @param oriented_king_square 已转换为观察方在下方的将帅位置。
/// @return 范围为 0～8 的九宫桶编号。
/// @throws std::invalid_argument 将帅位置不在下方九宫时抛出。
std::size_t king_bucket(int oriented_king_square) {
    const int row = square_row(oriented_king_square);
    const int col = square_col(oriented_king_square);
    if (row < 7 || row > 9 || col < 3 || col > 5) {
        throw std::invalid_argument("HalfKA perspective king must be inside its palace");
    }
    return static_cast<std::size_t>((row - 7) * 3 + (col - 3));
}

/// @brief 计算棋子相对于观察方的十四种 HalfKA 通道之一。
/// @param piece 要编码的非空棋子。
/// @param perspective 当前观察方。
/// @return 己方七种棋子使用 0～6，对方七种棋子使用 7～13。
std::size_t piece_channel(Piece piece, Color perspective) {
    const auto type = static_cast<std::size_t>(piece_type(piece));
    const std::size_t relative_color = piece_color(piece) == perspective ? 0U : 1U;
    return relative_color * 7U + (type - 1U);
}

}  // namespace

int orient_square_for_halfka(int square, Color perspective) {
    if (square < 0 || square >= kBoardSize) {
        throw std::out_of_range("HalfKA square outside board");
    }
    return perspective == Color::Red ? square : kBoardSize - 1 - square;
}

HalfKAFeatureIndex halfka_feature_index(
    const Position& position, Color perspective, int piece_square) {
    const Piece piece = position.piece_at(piece_square);
    if (is_empty(piece)) {
        throw std::invalid_argument("cannot encode an empty HalfKA feature");
    }

    const int king_square = position.king_square(perspective);
    if (king_square == kNoSquare) {
        throw std::invalid_argument("HalfKA perspective king is missing");
    }

    const std::size_t bucket = king_bucket(
        orient_square_for_halfka(king_square, perspective));
    const std::size_t channel = piece_channel(piece, perspective);
    const std::size_t square = static_cast<std::size_t>(
        orient_square_for_halfka(piece_square, perspective));
    return (bucket * kHalfKAPieceChannels + channel) * kBoardSize + square;
}

std::vector<HalfKAFeatureIndex> active_halfka_features(
    const Position& position, Color perspective) {
    std::vector<HalfKAFeatureIndex> features;
    features.reserve(32);
    for (int square = 0; square < kBoardSize; ++square) {
        if (!is_empty(position.piece_at(square))) {
            features.push_back(halfka_feature_index(position, perspective, square));
        }
    }
    std::sort(features.begin(), features.end());
    return features;
}

HalfKAInput extract_halfka_input(const Position& position) {
    return HalfKAInput{
        active_halfka_features(position, Color::Red),
        active_halfka_features(position, Color::Black),
    };
}

}  // namespace xiangqi
