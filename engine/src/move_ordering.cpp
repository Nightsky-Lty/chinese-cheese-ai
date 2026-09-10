#include "xiangqi/move_ordering.hpp"

#include "xiangqi/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xiangqi {
namespace {

constexpr int kTtMoveScore = 10'000'000;
constexpr int kGoodCaptureBase = 8'000'000;
constexpr int kQuietCheckScore = 7'000'000;
constexpr int kFirstKillerScore = 6'000'000;
constexpr int kSecondKillerScore = 5'900'000;
constexpr int kCounterMoveScore = 5'800'000;
constexpr int kQuietMoveBase = 1'000'000;
constexpr int kBadCaptureBase = -1'000'000;
constexpr int kCaptureCheckBonus = 50'000;
constexpr int kUnsafeCheckBonus = 1'000;
constexpr int kVictimValueScale = 32;
constexpr int kSeeValueScale = 64;
constexpr int kMaxExchangeDepth = 32;

std::size_t color_index(Color color) {
    return color == Color::Red ? 0U : 1U;
}

int attacker_selection_value(Piece piece) {
    if (piece_type(piece) == PieceType::King) {
        return 30'000;
    }
    return piece_base_value(piece_type(piece));
}

int exchange_gain(Position& position, int target, int exchange_depth) {
    if (exchange_depth >= kMaxExchangeDepth || is_empty(position.piece_at(target))) {
        return 0;
    }

    const std::vector<Move> legal_moves = position.generate_legal_moves();
    std::optional<Move> least_valuable_capture;
    int least_attacker_value = std::numeric_limits<int>::max();
    for (Move candidate : legal_moves) {
        if (candidate.to != target) {
            continue;
        }
        const Piece attacker = position.piece_at(candidate.from);
        const int attacker_value = attacker_selection_value(attacker);
        if (attacker_value < least_attacker_value) {
            least_attacker_value = attacker_value;
            least_valuable_capture = candidate;
        }
    }

    if (!least_valuable_capture) {
        return 0;
    }

    const int captured_value =
        piece_base_value(piece_type(position.piece_at(target)));
    const UndoInfo undo = position.do_move(*least_valuable_capture);
    const int reply_gain = exchange_gain(position, target, exchange_depth + 1);
    position.undo_move(*least_valuable_capture, undo);
    return std::max(0, captured_value - reply_gain);
}

struct ScoredMove {
    Move move{};
    int score{0};
};

}  // namespace

int capture_move_order_score(const Position& position, Move move) {
    const Piece victim = position.piece_at(move.to);
    if (is_empty(victim)) {
        return 0;
    }

    const Piece attacker = position.piece_at(move.from);
    return piece_base_value(piece_type(victim)) * kVictimValueScale -
           piece_base_value(piece_type(attacker));
}

void MoveOrdering::clear() {
    killers_ = {};
    history_ = {};
    piece_to_history_ = {};
    counter_moves_ = {};
}

bool MoveOrdering::gives_check(Position& position, Move move) {
    const UndoInfo undo = position.do_move(move);
    const bool result = position.in_check(position.side_to_move());
    position.undo_move(move, undo);
    return result;
}

int MoveOrdering::static_exchange_evaluation(Position& position, Move move) {
    const Piece victim = position.piece_at(move.to);
    const int initial_gain =
        is_empty(victim) ? 0 : piece_base_value(piece_type(victim));
    const UndoInfo undo = position.do_move(move);
    const int opponent_gain = exchange_gain(position, move.to, 1);
    position.undo_move(move, undo);
    return initial_gain - opponent_gain;
}

bool MoveOrdering::order_moves(
    Position& position, std::vector<Move>& moves,
    std::optional<Move> preferred_move, int ply,
    std::optional<PreviousMoveInfo> previous_move) const {
    std::optional<Move> counter_move;
    if (previous_move && !is_empty(previous_move->moved) &&
        previous_move->moved < kPieceCodeCount) {
        counter_move = counter_moves_[previous_move->moved][previous_move->move.to];
    }

    std::vector<ScoredMove> scored_moves;
    scored_moves.reserve(moves.size());
    bool preferred_found = false;

    for (Move move : moves) {
        if (preferred_move && move == *preferred_move) {
            scored_moves.push_back(ScoredMove{move, kTtMoveScore});
            preferred_found = true;
            continue;
        }

        const bool capture = !is_empty(position.piece_at(move.to));
        const bool check = gives_check(position, move);
        if (capture) {
            const int see = static_exchange_evaluation(position, move);
            const int tactical_score = see * kSeeValueScale +
                                       capture_move_order_score(position, move) +
                                       (check ? kCaptureCheckBonus : 0);
            scored_moves.push_back(ScoredMove{
                move,
                (see >= 0 ? kGoodCaptureBase : kBadCaptureBase) + tactical_score,
            });
            continue;
        }

        const int see = check ? static_exchange_evaluation(position, move) : 0;
        if (check && see >= 0) {
            scored_moves.push_back(ScoredMove{move, kQuietCheckScore + see});
            continue;
        }

        if (ply >= 0 && ply < kMaxMoveOrderingPly) {
            if (killers_[static_cast<std::size_t>(ply)][0] == move) {
                scored_moves.push_back(ScoredMove{move, kFirstKillerScore});
                continue;
            }
            if (killers_[static_cast<std::size_t>(ply)][1] == move) {
                scored_moves.push_back(ScoredMove{move, kSecondKillerScore});
                continue;
            }
        }

        if (counter_move && move == *counter_move) {
            scored_moves.push_back(ScoredMove{move, kCounterMoveScore});
            continue;
        }

        const Piece moving_piece = position.piece_at(move.from);
        const int main_history_score =
            history_[color_index(position.side_to_move())][move.from][move.to];
        const int piece_to_history_score =
            piece_to_history_[moving_piece][move.to];
        scored_moves.push_back(ScoredMove{
            move,
            kQuietMoveBase + main_history_score + piece_to_history_score +
                (check ? kUnsafeCheckBonus : 0),
        });
    }

    std::stable_sort(scored_moves.begin(), scored_moves.end(),
                     [](const ScoredMove& lhs, const ScoredMove& rhs) {
                         return lhs.score > rhs.score;
                     });
    std::transform(scored_moves.begin(), scored_moves.end(), moves.begin(),
                   [](const ScoredMove& scored) { return scored.move; });
    return preferred_found;
}

void MoveOrdering::record_quiet_beta_cutoff(
    const Position& position, Move move, int ply, int depth,
    std::optional<PreviousMoveInfo> previous_move,
    const std::vector<Move>& failed_quiet_moves) {
    if (ply >= 0 && ply < kMaxMoveOrderingPly) {
        auto& killers = killers_[static_cast<std::size_t>(ply)];
        if (killers[0] != move) {
            killers[1] = killers[0];
            killers[0] = move;
        }
    }

    if (previous_move && !is_empty(previous_move->moved) &&
        previous_move->moved < kPieceCodeCount) {
        counter_moves_[previous_move->moved][previous_move->move.to] = move;
    }

    const int bounded_depth = std::clamp(depth, 1, 64);
    const int bonus = bounded_depth * bounded_depth;
    auto& side_history = history_[color_index(position.side_to_move())];
    const Piece moving_piece = position.piece_at(move.from);
    update_history(side_history[move.from][move.to], bonus);
    update_history(piece_to_history_[moving_piece][move.to], bonus);
    for (Move failed : failed_quiet_moves) {
        if (failed != move) {
            update_history(side_history[failed.from][failed.to],
                           -std::max(1, bonus / 2));
            const Piece failed_piece = position.piece_at(failed.from);
            update_history(piece_to_history_[failed_piece][failed.to],
                           -std::max(1, bonus / 2));
        }
    }
}

void MoveOrdering::update_history(int& value, int bonus) {
    bonus = std::clamp(bonus, -kHistoryLimit, kHistoryLimit);
    value += bonus - value * std::abs(bonus) / kHistoryLimit;
}

}  // namespace xiangqi
