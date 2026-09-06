#include "xiangqi/game_history.hpp"

#include <algorithm>
#include <utility>

namespace xiangqi {

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

    const Color mover = position_.side_to_move();
    const Piece moved = position_.piece_at(move.from);
    const Piece captured = position_.piece_at(move.to);
    const std::uint64_t hash_before = position_.hash();
    const std::size_t previous_no_capture_plies = no_capture_plies_;
    const UndoInfo undo = position_.do_move(move);
    const bool gave_check = position_.in_check(position_.side_to_move());
    const std::uint64_t hash_after = position_.hash();

    no_capture_plies_ = is_empty(captured) ? no_capture_plies_ + 1 : 0;

    entries_.push_back(HistoryEntry{
        .move = move,
        .undo = undo,
        .mover = mover,
        .moved = moved,
        .captured = captured,
        .gave_check = gave_check,
        .previous_no_capture_plies = previous_no_capture_plies,
        .hash_before = hash_before,
        .hash_after = hash_after,
    });
    position_hashes_.push_back(hash_after);
    return true;
}

bool GameHistory::undo_last() {
    if (entries_.empty()) {
        return false;
    }

    const HistoryEntry entry = entries_.back();
    entries_.pop_back();
    position_hashes_.pop_back();
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
