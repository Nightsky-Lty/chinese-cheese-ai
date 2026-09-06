#pragma once

#include "xiangqi/position.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace xiangqi {

/// @brief 一条已经实际执行的对局记录。
struct HistoryEntry {
    Move move{};                              ///< 执行的走法。
    UndoInfo undo{};                          ///< 撤销走法所需的原状态。
    Color mover{Color::Red};                  ///< 执行走法的一方。
    Piece moved{kEmpty};                      ///< 移动的棋子。
    Piece captured{kEmpty};                   ///< 被吃棋子；未吃子时为 `kEmpty`。
    bool gave_check{false};                   ///< 该步完成后是否将军。
    std::size_t previous_no_capture_plies{0}; ///< 执行该步前的连续未吃子半回合数。
    std::uint64_t hash_before{0};             ///< 该步执行前的局面哈希。
    std::uint64_t hash_after{0};              ///< 该步执行后的局面哈希。
};

/// @brief 管理一盘实际对局的当前局面、走法记录、悔棋与重复局面信息。
///
/// `GameHistory` 只接受合法走法，并保留每一步是否吃子、是否将军等信息。
/// 重复检测只报告相同局面出现次数；中国象棋中的长将、长捉仍需由后续
/// 竞赛规则模块结合这些记录判定，不能简单地将三次重复直接判和。
class GameHistory {
public:
    /// @brief 使用标准初始局面创建一盘新对局。
    GameHistory();

    /// @brief 使用指定局面创建一盘新对局。
    /// @param initial_position 对局的起始局面。
    /// @param initial_no_capture_plies 起始局面已经连续未吃子的半回合数；载入残局时使用。
    explicit GameHistory(Position initial_position,
                         std::size_t initial_no_capture_plies = 0);

    /// @brief 查询当前局面。
    /// @return 当前局面的只读引用，避免绕过历史系统直接修改棋盘。
    [[nodiscard]] const Position& position() const { return position_; }

    /// @brief 查询已经执行的半回合数。
    /// @return 历史条目数量；双方各走一步计为两个半回合。
    [[nodiscard]] std::size_t ply_count() const { return entries_.size(); }

    /// @brief 查询当前是否存在可以撤销的走法。
    /// @return 至少执行过一步棋时返回 `true`。
    [[nodiscard]] bool can_undo() const { return !entries_.empty(); }

    /// @brief 查询全部走法记录。
    /// @return 按执行顺序排列的只读历史条目。
    [[nodiscard]] const std::vector<HistoryEntry>& entries() const { return entries_; }

    /// @brief 查询从起始局面到当前局面的完整哈希时间线。
    /// @return 只读哈希数组；第 0 项是起始局面，之后每个半回合对应一项。
    [[nodiscard]] const std::vector<std::uint64_t>& position_hashes() const {
        return position_hashes_;
    }

    /// @brief 查询当前连续未吃子的半回合数。
    /// @return 自最近一次吃子之后经过的半回合数。
    [[nodiscard]] std::size_t no_capture_plies() const { return no_capture_plies_; }

    /// @brief 查询最后一步走法记录。
    /// @return 有历史时返回最后一条记录，否则返回 `std::nullopt`。
    [[nodiscard]] std::optional<HistoryEntry> last_entry() const;

    /// @brief 尝试在当前局面执行一步棋并写入历史。
    /// @param move 要执行的走法。
    /// @return 走法合法且执行成功时返回 `true`；非法时不改变任何状态并返回 `false`。
    bool play(Move move);

    /// @brief 撤销最后一步棋。
    /// @return 成功撤销时返回 `true`；没有历史可撤销时返回 `false`。
    bool undo_last();

    /// @brief 将对局重置到新的起始局面并清空全部走法记录。
    /// @param initial_position 新的起始局面。
    /// @param initial_no_capture_plies 新局面已经连续未吃子的半回合数。
    void reset(Position initial_position, std::size_t initial_no_capture_plies = 0);

    /// @brief 统计当前局面在本盘历史中出现过多少次。
    /// @return 包含当前这次在内的出现次数；局面哈希同时包含行棋方。
    [[nodiscard]] std::size_t current_position_occurrences() const;

    /// @brief 判断当前局面是否至少重复了指定次数。
    /// @param required_occurrences 所需出现次数，必须大于 0。
    /// @return 达到指定次数时返回 `true`；参数为 0 时返回 `false`。
    [[nodiscard]] bool is_repetition(std::size_t required_occurrences = 3) const;

private:
    Position position_;
    std::vector<HistoryEntry> entries_;
    // 第一个元素是起始局面，之后每执行一步追加一个局面哈希。
    std::vector<std::uint64_t> position_hashes_;
    // 从最近一次吃子开始累计，供后续自然限着规则判定使用。
    std::size_t no_capture_plies_{0};
};

}  // namespace xiangqi
