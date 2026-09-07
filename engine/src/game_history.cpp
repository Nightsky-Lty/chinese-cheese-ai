#include "xiangqi/game_history.hpp"

#include <algorithm>
#include <utility>

namespace xiangqi {
namespace {

/// @brief 使用 SplitMix64 末轮对规则历史特征进行稳定混合。
/// @param value 待混合的 64 位输入。
/// @return 扩散后的 64 位值。
constexpr std::uint64_t mix_history(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

/// @brief 计算执行一步棋后的规则上下文哈希。
/// @param previous 上一个局面的规则上下文哈希。
/// @param entry 本步完整历史元数据。
/// @param no_capture_plies 本步完成后的连续未吃子半回合数。
/// @return 包含本步顺序和规则特征的新上下文哈希。
std::uint64_t next_rule_context_hash(std::uint64_t previous,
                                     const HistoryEntry& entry,
                                     std::size_t no_capture_plies) {
    std::uint64_t move_bits = static_cast<std::uint64_t>(entry.move.from) |
                              (static_cast<std::uint64_t>(entry.move.to) << 8U) |
                              (static_cast<std::uint64_t>(entry.moved) << 16U) |
                              (static_cast<std::uint64_t>(entry.captured) << 24U) |
                              (static_cast<std::uint64_t>(entry.gave_check) << 32U);
    move_bits ^= static_cast<std::uint64_t>(no_capture_plies) << 40U;
    return mix_history(previous ^ mix_history(entry.hash_after) ^ mix_history(move_bits));
}

}  // namespace

GameHistory::GameHistory()
    : GameHistory(Position::initial()) {}

GameHistory::GameHistory(Position initial_position,
                         std::size_t initial_no_capture_plies) {
    reset(std::move(initial_position), initial_no_capture_plies);
}

std::optional<HistoryEntry> GameHistory::last_entry() const {
    if (entries_.empty()) {
        return std::nullopt;
    }
    return entries_.back();
}

bool GameHistory::play(Move move) {
    if (!position_.is_legal_move(move)) {
        return false;
    }

    push_legal_move(move);
    return true;
}

void GameHistory::push_legal_move(Move move) {
    const Color mover = position_.side_to_move();
    const Piece moved = position_.piece_at(move.from);
    const Piece captured = position_.piece_at(move.to);
    const std::uint64_t hash_before = position_.hash();
    const std::size_t previous_no_capture_plies = no_capture_plies_;
    const UndoInfo undo = position_.do_move(move);
    const bool gave_check = position_.in_check(position_.side_to_move());
    const std::uint64_t hash_after = position_.hash();

    no_capture_plies_ = is_empty(captured) ? no_capture_plies_ + 1 : 0;

    const HistoryEntry entry{
        .move = move,
        .undo = undo,
        .mover = mover,
        .moved = moved,
        .captured = captured,
        .gave_check = gave_check,
        .previous_no_capture_plies = previous_no_capture_plies,
        .hash_before = hash_before,
        .hash_after = hash_after,
    };
    entries_.push_back(entry);
    position_hashes_.push_back(hash_after);
    rule_context_hashes_.push_back(
        next_rule_context_hash(rule_context_hashes_.back(), entry, no_capture_plies_));
}

bool GameHistory::undo_last() {
    if (entries_.empty()) {
        return false;
    }

    const HistoryEntry entry = entries_.back();
    entries_.pop_back();
    position_hashes_.pop_back();
    rule_context_hashes_.pop_back();
    position_.undo_move(entry.move, entry.undo);
    no_capture_plies_ = entry.previous_no_capture_plies;
    return true;
}

void GameHistory::reset(Position initial_position,
                        std::size_t initial_no_capture_plies) {
    position_ = std::move(initial_position);
    entries_.clear();
    position_hashes_.clear();
    position_hashes_.push_back(position_.hash());
    no_capture_plies_ = initial_no_capture_plies;
    rule_context_hashes_.clear();
    rule_context_hashes_.push_back(mix_history(
        position_.hash() ^ mix_history(static_cast<std::uint64_t>(no_capture_plies_))));
}

std::size_t GameHistory::current_position_occurrences() const {
    const std::uint64_t current_hash = position_.hash();
    return static_cast<std::size_t>(
        std::count(position_hashes_.begin(), position_hashes_.end(), current_hash));
}

bool GameHistory::is_repetition(std::size_t required_occurrences) const {
    return required_occurrences != 0 &&
           current_position_occurrences() >= required_occurrences;
}

}  // namespace xiangqi
