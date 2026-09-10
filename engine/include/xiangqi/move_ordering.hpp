#pragma once

#include "xiangqi/position.hpp"

#include <array>
#include <optional>
#include <vector>

namespace xiangqi {

constexpr int kMaxMoveOrderingPly = 256;

/// @brief 描述当前节点的上一手棋，供 Counter Move 启发查询。
struct PreviousMoveInfo {
    Move move{};          ///< 上一手的起点和终点。
    Piece moved{kEmpty};  ///< 上一手移动的棋子编码。
};

/// @brief 计算走法的基础吃子排序分（MVV-LVA）。
/// @param position 走法尚未执行时的父局面。
/// @param move 待评分走法；调用者应传入当前局面的合法走法。
/// @return 非吃子返回 0；吃子优先吃高价值棋子，同一目标下优先使用低价值棋子。
/// @note 棋子价值直接来自 `piece_base_value`，与手工评估器保持一致。
[[nodiscard]] int capture_move_order_score(const Position& position, Move move);

/// @brief 保存并应用搜索过程中积累的走法排序启发。
class MoveOrdering {
public:
    /// @brief 清空 Killer、Counter Move、Main History 和 Piece-to History 启发。
    void clear();

    /// @brief 判断合法走法执行后是否会将军，并在返回前恢复局面。
    /// @param position 走法执行前的局面。
    /// @param move 当前局面中的合法走法。
    /// @return 走后对方将帅受到攻击时返回 `true`。
    [[nodiscard]] static bool gives_check(Position& position, Move move);

    /// @brief 静态估算目标格上连续吃子和反吃的最终子力收益。
    /// @param position 走法执行前的局面；返回前会完整恢复。
    /// @param move 要评估的合法走法，既可以是吃子也可以是安静着。
    /// @return 当前行棋方的估算净收益；正数获利，0 大致安全，负数亏损。
    [[nodiscard]] static int static_exchange_evaluation(Position& position, Move move);

    /// @brief 按 TT、有利吃子、安全将军、Killer、Counter、History、亏损吃子排序。
    /// @param position 生成这些走法的父局面；评分期间的试走会被完整撤销。
    /// @param moves 要原地排序的合法走法列表。
    /// @param preferred_move 可选的 TT 或上一轮 PV 最佳着，若合法则置于首位。
    /// @param ply 当前节点距根节点的半回合数，用于查询 Killer。
    /// @param previous_move 当前节点上一手棋的信息，用于查询 Counter Move。
    /// @return `preferred_move` 存在于列表并成功置顶时返回 `true`。
    bool order_moves(Position& position, std::vector<Move>& moves,
                     std::optional<Move> preferred_move, int ply,
                     std::optional<PreviousMoveInfo> previous_move) const;

    /// @brief 记录一次由安静着造成的 Beta 截断并更新各类启发。
    /// @param position 截断走法执行前的父局面，用于取得行棋方和移动棋子。
    /// @param move 导致 Beta 截断的安静着。
    /// @param ply 当前节点距根节点的半回合数。
    /// @param depth 当前节点剩余搜索深度，越深的成功获得越大奖励。
    /// @param previous_move 对手上一手棋，用于写入 Counter Move。
    /// @param failed_quiet_moves 本节点在截断前搜索过但失败的安静着。
    void record_quiet_beta_cutoff(const Position& position, Move move,
                                  int ply, int depth,
                                  std::optional<PreviousMoveInfo> previous_move,
                                  const std::vector<Move>& failed_quiet_moves);

private:
    static constexpr std::size_t kPieceCodeCount = 16;
    static constexpr int kHistoryLimit = 16'384;

    std::array<std::array<std::optional<Move>, 2>, kMaxMoveOrderingPly> killers_{};
    std::array<std::array<std::array<int, kBoardSize>, kBoardSize>, 2> history_{};
    // 棋子编码同时包含阵营和类型；该表跨起点学习“某类棋子走到某格”的经验。
    std::array<std::array<int, kBoardSize>, kPieceCodeCount> piece_to_history_{};
    std::array<std::array<std::optional<Move>, kBoardSize>, kPieceCodeCount>
        counter_moves_{};

    /// @brief 使用有界公式更新历史分，避免无限增长和整数溢出。
    /// @param value 要原地更新的历史分数。
    /// @param bonus 正值为奖励，负值为惩罚。
    static void update_history(int& value, int bonus);
};

}  // namespace xiangqi
