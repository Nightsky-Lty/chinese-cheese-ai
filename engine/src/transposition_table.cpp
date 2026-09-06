#include "xiangqi/transposition_table.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace xiangqi {

TranspositionTable::TranspositionTable(std::size_t megabytes) {
    resize(megabytes);
}

void TranspositionTable::resize(std::size_t megabytes) {
    constexpr std::size_t bytes_per_megabyte = 1024U * 1024U;
    const std::size_t max_megabytes =
        std::numeric_limits<std::size_t>::max() / bytes_per_megabyte;
    const std::size_t safe_megabytes = std::min(megabytes, max_megabytes);
    const std::size_t target_bytes = safe_megabytes * bytes_per_megabyte;
    const std::size_t requested_entries =
        std::max<std::size_t>(1, target_bytes / sizeof(TTEntry));
    const std::size_t power_of_two_entries = std::bit_floor(requested_entries);

    entries_.assign(power_of_two_entries, TTEntry{});
    index_mask_ = power_of_two_entries - 1;
    generation_ = 0;
}

void TranspositionTable::clear() {
    std::fill(entries_.begin(), entries_.end(), TTEntry{});
    generation_ = 0;
}

void TranspositionTable::new_search() {
    ++generation_;
    if (generation_ == 0) {
        // 代数回绕时清空，避免 256 次搜索前的旧条目被误认为当前代。
        clear();
        generation_ = 1;
    }
}

const TTEntry* TranspositionTable::probe(std::uint64_t key) const {
    const TTEntry& entry = entries_[static_cast<std::size_t>(key) & index_mask_];
    return entry.valid && entry.key == key ? &entry : nullptr;
}

void TranspositionTable::store(std::uint64_t key, int depth, int score,
                               TTBound bound, Move best_move) {
    TTEntry& current = entries_[static_cast<std::size_t>(key) & index_mask_];

    bool replace = !current.valid;
    if (current.valid && current.key == key) {
        // 同一局面优先保留搜索更深的结果。
        replace = depth >= current.depth;
    } else if (current.valid) {
        // 哈希冲突时优先替换旧代条目，其次替换不更深的条目。
        replace = current.generation != generation_ || depth >= current.depth;
    }

    if (!replace) {
        return;
    }
    current = TTEntry{
        .key = key,
        .best_move = best_move,
        .score = score,
        .depth = depth,
        .bound = bound,
        .generation = generation_,
        .valid = true,
    };
}

}  // namespace xiangqi
