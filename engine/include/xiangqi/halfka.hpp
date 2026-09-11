#pragma once

#include "xiangqi/position.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace xiangqi {

using HalfKAFeatureIndex = std::size_t;

constexpr std::size_t kHalfKAKingBuckets = 9;
constexpr std::size_t kHalfKAPieceChannels = 14;
constexpr std::size_t kHalfKAFeatureDimensions =
    kHalfKAKingBuckets * kHalfKAPieceChannels * kBoardSize;

/// @brief 保存红方和黑方两个视角下激活的 HalfKA 稀疏特征。
/// @note 数组下标与 `Color` 的底层值一致：0 为红方视角，1 为黑方视角。
using HalfKAInput = std::array<std::vector<HalfKAFeatureIndex>, 2>;

/// @brief 把棋盘位置转换到指定一方的标准观察方向。
/// @param square 原始棋盘下标，范围为 0～89。
/// @param perspective 观察局面的阵营；黑方视角会把棋盘旋转 180 度。
/// @return 变换后的位置，使观察方的九宫始终位于棋盘下方。
/// @throws std::out_of_range `square` 不在棋盘范围内时抛出。
[[nodiscard]] int orient_square_for_halfka(int square, Color perspective);

/// @brief 计算一个非空棋子在指定视角下对应的 HalfKA 特征编号。
/// @param position 提供观察方将帅位置的有效局面。
/// @param perspective 当前特征累加器所属的观察方。
/// @param piece_square 要编码的非空棋子所在位置。
/// @return 范围为 `[0, kHalfKAFeatureDimensions)` 的唯一特征编号。
/// @throws std::out_of_range `piece_square` 越界时抛出。
/// @throws std::invalid_argument 目标位置为空，或观察方将帅不在己方九宫时抛出。
[[nodiscard]] HalfKAFeatureIndex halfka_feature_index(
    const Position& position, Color perspective, int piece_square);

/// @brief 使用显式棋子和将帅位置计算 HalfKA 特征编号。
/// @param perspective_king_square 观察方将帅的原始棋盘位置。
/// @param perspective 当前特征累加器所属的观察方。
/// @param piece 要编码的非空棋子。
/// @param piece_square 该棋子所在的原始棋盘位置。
/// @return 范围为 `[0, kHalfKAFeatureDimensions)` 的唯一特征编号。
/// @throws std::out_of_range 任一位置越界时抛出。
/// @throws std::invalid_argument 棋子为空或观察方将帅不在己方九宫时抛出。
[[nodiscard]] HalfKAFeatureIndex halfka_feature_index(
    int perspective_king_square, Color perspective,
    Piece piece, int piece_square);

/// @brief 提取一个视角下当前局面所有激活的 HalfKA 稀疏特征。
/// @param position 要编码的有效象棋局面。
/// @param perspective 当前特征累加器所属的观察方。
/// @return 每个在盘棋子对应一个编号的稀疏特征列表，其中包括双方将帅。
[[nodiscard]] std::vector<HalfKAFeatureIndex> active_halfka_features(
    const Position& position, Color perspective);

/// @brief 同时提取红方和黑方两个视角下的 HalfKA 稀疏输入。
/// @param position 要编码的有效象棋局面。
/// @return 下标 0 为红方视角、下标 1 为黑方视角的特征列表。
[[nodiscard]] HalfKAInput extract_halfka_input(const Position& position);

}  // namespace xiangqi
