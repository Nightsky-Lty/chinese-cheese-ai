#pragma once

#include "xiangqi/position.hpp"

#include <array>
#include <string_view>

namespace xiangqi {

/// @brief 手工评估的分项结果，分数均以红方减黑方表示。
struct EvaluationBreakdown {
    int material{0};    ///< 基础子力差。
    int positional{0};  ///< 棋子位置、过河和中心控制奖励之差。

    /// @brief 计算红方视角的总评估分。
    /// @return `material + positional`；正数表示红方有利。
    [[nodiscard]] int total() const { return material + positional; }
};

/// @brief 定义搜索器可使用的统一局面评估接口。
/// @note 实现类只负责非终局的静态评分；将死、重复和长将等结果由规则与搜索层处理。
class Evaluator {
public:
    virtual ~Evaluator() = default;

    /// @brief 返回评估器的人类可读名称。
    /// @return 生命周期不短于评估器对象的名称字符串。
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// @brief 从指定阵营视角评估当前局面。
    /// @param position 待评估的非终局局面。
    /// @param perspective 评分所站的阵营。
    /// @return 正数表示 `perspective` 有利，负数表示其不利。
    [[nodiscard]] virtual int evaluate_for(
        const Position& position, Color perspective) const = 0;

    /// @brief 从当前行棋方视角评估局面，供 Negamax 搜索直接调用。
    /// @param position 待评估的非终局局面。
    /// @return 正数表示当前行棋方有利，负数表示当前行棋方不利。
    [[nodiscard]] int evaluate(const Position& position) const {
        return evaluate_for(position, position.side_to_move());
    }
};

/// @brief 使用子力和轻量位置奖励进行评分的手工评估器。
class HandcraftedEvaluator final : public Evaluator {
public:
    /// @brief 返回手工评估器的名称。
    /// @return 固定字符串 `handcrafted`。
    [[nodiscard]] std::string_view name() const noexcept override;

    /// @brief 从指定阵营视角执行手工局面评估。
    /// @param position 待评估局面。
    /// @param perspective 评分所站的阵营。
    /// @return 正数表示 `perspective` 有利，负数表示其不利。
    [[nodiscard]] int evaluate_for(
        const Position& position, Color perspective) const override;

    /// @brief 分项计算子力和位置评分，便于调试与展示。
    /// @param position 待评估局面。
    /// @return 以红方减黑方表示的手工评估分项。
    [[nodiscard]] EvaluationBreakdown breakdown(const Position& position) const;
};

/// @brief 取得进程内共享的无状态手工评估器。
/// @return 可安全长期引用的 `HandcraftedEvaluator` 单例。
[[nodiscard]] const HandcraftedEvaluator& handcrafted_evaluator();

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
