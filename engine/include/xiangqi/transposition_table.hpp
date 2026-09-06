#pragma once

#include "xiangqi/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xiangqi {

/// @brief 置换表分数的边界类型。
enum class TTBound : std::uint8_t {
    Exact, ///< 分数位于完整 Alpha-Beta 窗口内，是精确值。
    Lower, ///< 分数至少为保存值，通常来自 Beta 截断。
    Upper, ///< 分数至多为保存值，通常未能提高 Alpha。
};

/// @brief 置换表中的单个局面缓存条目。
struct TTEntry {
    std::uint64_t key{0};          ///< 完整的 Zobrist 局面哈希。
    Move best_move{};              ///< 该局面已知的最佳走法。
    int score{0};                  ///< 经将死距离归一化后保存的搜索分数。
    int depth{-1};                 ///< 产生该分数的剩余搜索深度。
    TTBound bound{TTBound::Exact}; ///< 分数是精确值、下界还是上界。
    std::uint8_t generation{0};    ///< 写入条目时的搜索代数。
    bool valid{false};             ///< 该槽位是否包含有效条目。
};

/// @brief 固定容量、单槽替换的 Alpha-Beta 置换表。
///
/// 表容量始终取不超过目标内存的 2 的幂，索引可通过位掩码快速计算。
/// 哈希冲突只会降低缓存命中率，不影响搜索正确性，因为命中时会核对完整键。
class TranspositionTable {
public:
    /// @brief 创建指定容量的置换表。
    /// @param megabytes 目标内存容量，0 会退化为至少一个条目的表。
    explicit TranspositionTable(std::size_t megabytes = 16);

    /// @brief 重新设置容量并清空全部缓存。
    /// @param megabytes 新的目标内存容量。
    void resize(std::size_t megabytes);

    /// @brief 清空所有条目并重置搜索代数。
    void clear();

    /// @brief 开始新的一次根节点搜索，使新条目获得新的代数。
    void new_search();

    /// @brief 查询与哈希完全匹配的缓存条目。
    /// @param key 当前局面的完整 Zobrist 哈希。
    /// @return 命中时返回只读条目指针，否则返回 `nullptr`。
    [[nodiscard]] const TTEntry* probe(std::uint64_t key) const;

    /// @brief 按深度和代数替换策略保存一个搜索结果。
    /// @param key 当前局面的完整 Zobrist 哈希。
    /// @param depth 产生该结果的剩余搜索深度。
    /// @param score 已完成将死距离归一化的分数。
    /// @param bound 分数边界类型。
    /// @param best_move 当前局面的最佳走法。
    void store(std::uint64_t key, int depth, int score, TTBound bound, Move best_move);

    /// @brief 查询置换表可以容纳的条目数量。
    /// @return 当前槽位总数。
    [[nodiscard]] std::size_t entry_count() const { return entries_.size(); }

private:
    std::vector<TTEntry> entries_;
    std::size_t index_mask_{0};
    std::uint8_t generation_{0};
};

}  // namespace xiangqi
