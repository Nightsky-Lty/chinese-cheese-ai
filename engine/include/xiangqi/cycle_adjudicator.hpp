#pragma once

#include "xiangqi/game_history.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace xiangqi {

/// @brief 触发重复裁决所需的相同局面出现次数。
constexpr std::size_t kRepetitionOccurrences = 3;

/// @brief 触发自然限着和棋的连续未吃子半回合数，即 60 个整回合。
constexpr std::size_t kNoCaptureDrawPlies = 120;

/// @brief 当前局面最近若干次出现的位置。
struct RepetitionWindow {
    /// 局面在哈希时间线中的半回合边界下标，按时间升序排列。
    std::vector<std::size_t> occurrence_plies;

    /// @brief 查询本次重复判定覆盖的第一个局面边界。
    /// @return `occurrence_plies` 的第一项。
    [[nodiscard]] std::size_t begin_ply() const { return occurrence_plies.front(); }

    /// @brief 查询当前局面所在的半回合边界。
    /// @return `occurrence_plies` 的最后一项。
    [[nodiscard]] std::size_t end_ply() const { return occurrence_plies.back(); }
};

/// @brief 三次重复中的长将与长捉分析结果。
enum class RepetitionVerdict {
    None,                 ///< 当前局面尚未达到指定重复次数。
    Draw,                 ///< 达到重复次数，但没有可归责于单方的长将或长捉。
    RedPerpetualCheck,    ///< 红方在重复区间内自己的每一步均为将军，红方判负。
    BlackPerpetualCheck,  ///< 黑方在重复区间内自己的每一步均为将军，黑方判负。
    RedPerpetualChase,    ///< 红方在重复区间内每一步都新捉同一棋子，红方判负。
    BlackPerpetualChase,  ///< 黑方在重复区间内每一步都新捉同一棋子，黑方判负。
};

/// @brief 重复区间及其裁决结果。
struct RepetitionAdjudication {
    RepetitionVerdict verdict{RepetitionVerdict::None};
    std::optional<RepetitionWindow> window;
};

/// @brief 一盘对局基于历史信息得到的自动裁决结果。
enum class HistoryVerdict {
    Ongoing,                      ///< 对局继续。
    DrawByThreefoldRepetition,    ///< 同一局面至少出现三次，且没有单方长将或长捉。
    DrawByNoCapture,              ///< 连续 120 个半回合没有吃子。
    RedWinsByBlackPerpetualCheck, ///< 黑方长将判负，因此红方获胜。
    BlackWinsByRedPerpetualCheck, ///< 红方长将判负，因此黑方获胜。
    RedWinsByBlackPerpetualChase, ///< 黑方长捉判负，因此红方获胜。
    BlackWinsByRedPerpetualChase, ///< 红方长捉判负，因此黑方获胜。
};

/// @brief 查找当前局面最近的若干次出现位置。
/// @param history 要查询的完整对局历史。
/// @param required_occurrences 所需出现次数，默认三次且必须至少为 2。
/// @return 次数足够时返回最近的出现位置，否则返回空。
/// @note 局面哈希包含行棋方，因此棋子位置相同但行棋方不同不会被视为相同局面。
[[nodiscard]] std::optional<RepetitionWindow> find_repetition_window(
    const GameHistory& history,
    std::size_t required_occurrences = kRepetitionOccurrences);

/// @brief 按“三次重复判和、单方长将或长捉判负”规则裁决当前重复局面。
/// @param history 要分析的完整对局历史。
/// @param required_occurrences 触发裁决的相同局面出现次数，默认三次且必须至少为 2。
/// @return 重复区间和裁决；尚未达到次数时返回 `None`。
[[nodiscard]] RepetitionAdjudication adjudicate_repetition(
    const GameHistory& history,
    std::size_t required_occurrences = kRepetitionOccurrences);

/// @brief 综合三次重复和 60 回合未吃子规则裁决当前对局。
/// @param history 要裁决的完整对局历史。
/// @return 胜、和或继续；重复长将、长捉优先于自然限着和棋。
[[nodiscard]] HistoryVerdict adjudicate_history(const GameHistory& history);

}  // namespace xiangqi
