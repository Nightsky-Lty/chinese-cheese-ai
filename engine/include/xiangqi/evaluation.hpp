#pragma once

#include "xiangqi/position.hpp"

#include <array>

namespace xiangqi {

/// @brief 手工评估的分项结果，分数均以红方减黑方表示。
struct EvaluationBreakdown {
    int material{0};    ///< 基础子力差。
    int positional{0};  ///< 棋子位置、过河和中心控制奖励之差。

    /// @brief 计算红方视角的总评估分。
    /// @return `material + positional`；正数表示红方有利。
    [[nodiscard]] int total() const { return material + positional; }
};

/// @brief 查询一种棋子的基础子力价值。
/// @param type 棋子类型。
/// @return 以一个未过河兵约等于 100 分为尺度的基础价值；将帅和空位置返回 0。
[[nodiscard]] int piece_base_value(PieceType type);

/// @brief 分项计算当前局面的手工评估。
/// @param position 待评估局面。
/// @return 红方减黑方视角的子力和位置分项，不处理将死与重复判罚。
[[nodiscard]] EvaluationBreakdown evaluate_breakdown(const Position& position);

/// @brief 从指定阵营视角评估当前局面。
/// @param position 待评估局面。
/// @param perspective 评分所站的阵营。
/// @return 正数表示 `perspective` 有利，负数表示其不利。
[[nodiscard]] int evaluate_for(const Position& position, Color perspective);

/// @brief 从当前行棋方视角评估局面，供 Negamax 搜索直接调用。
/// @param position 待评估局面。
/// @return 正数表示当前行棋方有利，负数表示当前行棋方不利。
/// @note 将死、困毙、重复和长将等确定性结果应由搜索或规则层处理。
[[nodiscard]] int evaluate(const Position& position);

}  // namespace xiangqi
