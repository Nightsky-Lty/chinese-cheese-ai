#include "xiangqi/search.hpp"

#include "xiangqi/cycle_adjudicator.hpp"
#include "xiangqi/move_ordering.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace xiangqi {
namespace {

/// @brief 将无合法走法的节点转换为当前行棋方的失败分数。
/// @param ply 当前节点距离根节点的半回合数。
/// @return 带将死距离的负分；越早失败，绝对值越大。
int terminal_loss_score(int ply) {
    return -kMateScore + ply;
}

/// @brief 从对局历史提取 Counter Move 所需的上一手信息。
/// @param history 当前节点的完整历史。
/// @return 有上一手时返回其走法和移动棋子，否则返回空。
std::optional<PreviousMoveInfo> previous_move_info(const GameHistory& history) {
    const std::optional<HistoryEntry> entry = history.last_entry();
    if (!entry) {
        return std::nullopt;
    }
    return PreviousMoveInfo{entry->move, entry->moved};
}

}  // namespace

Searcher::Searcher(std::size_t transposition_table_mb)
    : Searcher(handcrafted_evaluator(), transposition_table_mb) {}

Searcher::Searcher(const Evaluator& evaluator, std::size_t transposition_table_mb)
    : evaluator_(evaluator), transposition_table_(transposition_table_mb) {}

void Searcher::clear_transposition_table() {
    transposition_table_.clear();
    cached_quiescence_depth_.reset();
}

SearchResult Searcher::search(Position& position, const SearchLimits& limits) {
    GameHistory history(position);
    return search(history, limits);
}

SearchResult Searcher::search(GameHistory& history, const SearchLimits& limits) {
    if (limits.depth < 0) {
        throw std::invalid_argument("search depth cannot be negative");
    }
    if (limits.quiescence_depth < 0) {
        throw std::invalid_argument("quiescence depth cannot be negative");
    }

    evaluation_state_ = evaluator_.create_state(history.position_);
    if (!evaluation_state_) {
        throw std::logic_error("evaluator returned a null search state");
    }

    nodes_ = 0;
    beta_cutoffs_ = 0;
    quiescence_nodes_ = 0;
    tt_hits_ = 0;
    tt_cutoffs_ = 0;
    tt_move_orderings_ = 0;
    pvs_researches_ = 0;
    move_ordering_.clear();
    quiescence_depth_ = limits.quiescence_depth;
    use_transposition_table_ =
        limits.use_transposition_table && limits.algorithm == SearchAlgorithm::AlphaBeta;
    if (use_transposition_table_) {
        if (cached_quiescence_depth_ &&
            *cached_quiescence_depth_ != limits.quiescence_depth) {
            transposition_table_.clear();
        }
        cached_quiescence_depth_ = limits.quiescence_depth;
        transposition_table_.new_search();
    }

    SearchResult result;
    if (limits.depth == 0) {
        const std::uint64_t nodes_before = nodes_;
        const std::uint64_t cutoffs_before = beta_cutoffs_;
        const std::uint64_t qnodes_before = quiescence_nodes_;
        const std::uint64_t tt_hits_before = tt_hits_;
        const std::uint64_t tt_cutoffs_before = tt_cutoffs_;
        const std::uint64_t tt_move_orderings_before = tt_move_orderings_;
        const std::uint64_t pvs_researches_before = pvs_researches_;
        result = search_iteration(history, 0, limits.algorithm, std::nullopt);
        result.iterations.push_back(IterationResult{
            .depth = 0,
            .best_move = result.best_move,
            .score = result.score,
            .nodes = nodes_ - nodes_before,
            .beta_cutoffs = beta_cutoffs_ - cutoffs_before,
            .quiescence_nodes = quiescence_nodes_ - qnodes_before,
            .tt_hits = tt_hits_ - tt_hits_before,
            .tt_cutoffs = tt_cutoffs_ - tt_cutoffs_before,
            .tt_move_orderings = tt_move_orderings_ - tt_move_orderings_before,
            .pvs_researches = pvs_researches_ - pvs_researches_before,
        });
        result.nodes = nodes_;
        result.beta_cutoffs = beta_cutoffs_;
        result.quiescence_nodes = quiescence_nodes_;
        result.tt_hits = tt_hits_;
        result.tt_cutoffs = tt_cutoffs_;
        result.tt_move_orderings = tt_move_orderings_;
        result.pvs_researches = pvs_researches_;
        return result;
    }

    std::optional<Move> previous_best;
    for (int current_depth = 1; current_depth <= limits.depth; ++current_depth) {
        const std::uint64_t nodes_before = nodes_;
        const std::uint64_t cutoffs_before = beta_cutoffs_;
        const std::uint64_t qnodes_before = quiescence_nodes_;
        const std::uint64_t tt_hits_before = tt_hits_;
        const std::uint64_t tt_cutoffs_before = tt_cutoffs_;
        const std::uint64_t tt_move_orderings_before = tt_move_orderings_;
        const std::uint64_t pvs_researches_before = pvs_researches_;

        SearchResult iteration = search_iteration(
            history, current_depth, limits.algorithm, previous_best);
        result.best_move = iteration.best_move;
        result.score = iteration.score;
        result.depth = current_depth;
        result.iterations.push_back(IterationResult{
            .depth = current_depth,
            .best_move = iteration.best_move,
            .score = iteration.score,
            .nodes = nodes_ - nodes_before,
            .beta_cutoffs = beta_cutoffs_ - cutoffs_before,
            .quiescence_nodes = quiescence_nodes_ - qnodes_before,
            .tt_hits = tt_hits_ - tt_hits_before,
            .tt_cutoffs = tt_cutoffs_ - tt_cutoffs_before,
            .tt_move_orderings = tt_move_orderings_ - tt_move_orderings_before,
            .pvs_researches = pvs_researches_ - pvs_researches_before,
        });

        if (!iteration.best_move) {
            break;
        }
        previous_best = iteration.best_move;
    }

    result.nodes = nodes_;
    result.beta_cutoffs = beta_cutoffs_;
    result.quiescence_nodes = quiescence_nodes_;
    result.tt_hits = tt_hits_;
    result.tt_cutoffs = tt_cutoffs_;
    result.tt_move_orderings = tt_move_orderings_;
    result.pvs_researches = pvs_researches_;
    return result;
}

void Searcher::push_search_move(GameHistory& history, Move move) {
    history.push_legal_move(move);
    const HistoryEntry& entry = history.entries_.back();
    try {
        evaluation_state_->push_move(
            history.position_, move, entry.moved, entry.captured);
    } catch (...) {
        static_cast<void>(history.undo_last());
        throw;
    }
}

void Searcher::pop_search_move(GameHistory& history) {
    if (!history.can_undo()) {
        throw std::logic_error("cannot pop an empty search path");
    }
    evaluation_state_->pop_move();
    if (!history.undo_last()) {
        throw std::logic_error("failed to undo synchronized search move");
    }
}

SearchResult Searcher::search_iteration(GameHistory& history, int depth,
                                        SearchAlgorithm algorithm,
                                        std::optional<Move> previous_best) {
    ++nodes_;
    SearchResult result;
    result.depth = depth;
    Position& position = history.position_;

    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        result.score = terminal_loss_score(0);
        return result;
    }
    if (const std::optional<int> score = adjudication_score(history, 0)) {
        result.score = *score;
        return result;
    }
    if (depth == 0) {
        result.score = quiescence(
            history, -kSearchInfinity, kSearchInfinity, 0, 0);
        return result;
    }

    std::optional<Move> preferred_move = previous_best;
    bool preferred_move_from_tt = false;
    if (use_transposition_table_) {
        if (const TTEntry* entry = transposition_table_.probe(transposition_key(history))) {
            ++tt_hits_;
            if (!preferred_move) {
                preferred_move = entry->best_move;
                preferred_move_from_tt = true;
            }
        }
    }
    if (move_ordering_.order_moves(position, moves, preferred_move, 0,
                                   previous_move_info(history)) &&
        preferred_move_from_tt) {
        ++tt_move_orderings_;
    }

    int best_score = -kSearchInfinity;
    Move best_move = moves.front();
    int alpha = -kSearchInfinity;
    constexpr int beta = kSearchInfinity;

    bool first_move = true;
    for (Move move : moves) {
        push_search_move(history, move);
        int score = 0;
        if (algorithm == SearchAlgorithm::Negamax) {
            score = -negamax(history, depth - 1, 1);
        } else if (first_move) {
            // 主变化候选使用完整窗口，建立当前根节点的可靠 Alpha。
            score = -alpha_beta(history, depth - 1, -beta, -alpha, 1);
        } else {
            // 后续着先用零窗口判断是否能够超过当前最佳分数。
            score = -alpha_beta(history, depth - 1, -alpha - 1, -alpha, 1);
            if (score > alpha && score < beta) {
                ++pvs_researches_;
                score = -alpha_beta(history, depth - 1, -beta, -alpha, 1);
            }
        }
        pop_search_move(history);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (algorithm == SearchAlgorithm::AlphaBeta) {
            alpha = std::max(alpha, score);
        }
        first_move = false;
    }

    result.best_move = best_move;
    result.score = best_score;
    if (use_transposition_table_) {
        transposition_table_.store(transposition_key(history), depth,
                                   score_to_tt(best_score, 0),
                                   TTBound::Exact, best_move);
    }
    return result;
}

int Searcher::negamax(GameHistory& history, int depth, int ply) {
    ++nodes_;
    Position& position = history.position_;
    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }
    if (const std::optional<int> score = adjudication_score(history, ply)) {
        return *score;
    }
    if (depth == 0) {
        return quiescence(history, -kSearchInfinity, kSearchInfinity, ply, 0);
    }

    move_ordering_.order_moves(position, moves, std::nullopt, ply,
                               previous_move_info(history));
    int best_score = -kSearchInfinity;
    for (Move move : moves) {
        push_search_move(history, move);
        const int score = -negamax(history, depth - 1, ply + 1);
        pop_search_move(history);
        best_score = std::max(best_score, score);
    }
    return best_score;
}

int Searcher::alpha_beta(GameHistory& history, int depth, int alpha, int beta, int ply) {
    ++nodes_;
    Position& position = history.position_;
    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }
    if (const std::optional<int> score = adjudication_score(history, ply)) {
        return *score;
    }
    if (depth == 0) {
        return quiescence(history, alpha, beta, ply, 0);
    }

    const int original_alpha = alpha;
    std::optional<Move> tt_move;
    if (use_transposition_table_) {
        if (const TTEntry* entry = transposition_table_.probe(transposition_key(history))) {
            ++tt_hits_;
            tt_move = entry->best_move;
            if (entry->depth >= depth) {
                const int tt_score = score_from_tt(entry->score, ply);
                if (entry->bound == TTBound::Exact ||
                    (entry->bound == TTBound::Lower && tt_score >= beta) ||
                    (entry->bound == TTBound::Upper && tt_score <= alpha)) {
                    ++tt_cutoffs_;
                    return tt_score;
                }
            }
        }
    }

    const std::optional<PreviousMoveInfo> previous_move = previous_move_info(history);
    if (move_ordering_.order_moves(position, moves, tt_move, ply, previous_move) &&
        tt_move) {
        ++tt_move_orderings_;
    }
    int best_score = -kSearchInfinity;
    Move best_move = moves.front();
    std::vector<Move> failed_quiet_moves;
    bool first_move = true;
    for (Move move : moves) {
        const bool quiet = is_empty(position.piece_at(move.to));
        push_search_move(history, move);
        int score = 0;
        if (first_move) {
            score = -alpha_beta(history, depth - 1, -beta, -alpha, ply + 1);
        } else {
            score = -alpha_beta(history, depth - 1, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta) {
                ++pvs_researches_;
                score = -alpha_beta(history, depth - 1, -beta, -alpha, ply + 1);
            }
        }
        pop_search_move(history);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++beta_cutoffs_;
            if (quiet) {
                move_ordering_.record_quiet_beta_cutoff(
                    position, move, ply, depth,
                    previous_move, failed_quiet_moves);
            }
            break;
        }
        if (quiet) {
            failed_quiet_moves.push_back(move);
        }
        first_move = false;
    }

    if (use_transposition_table_) {
        TTBound bound = TTBound::Exact;
        if (best_score <= original_alpha) {
            bound = TTBound::Upper;
        } else if (best_score >= beta) {
            bound = TTBound::Lower;
        }
        transposition_table_.store(transposition_key(history), depth,
                                   score_to_tt(best_score, ply), bound, best_move);
    }
    return best_score;
}

int Searcher::quiescence(GameHistory& history, int alpha, int beta, int ply, int qply) {
    ++quiescence_nodes_;
    Position& position = history.position_;

    const bool checked = position.in_check(position.side_to_move());
    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }
    if (const std::optional<int> score = adjudication_score(history, ply)) {
        return *score;
    }

    const int stand_pat = evaluation_state_->evaluate(position);
    if (qply >= quiescence_depth_) {
        return stand_pat;
    }

    if (!checked) {
        if (stand_pat >= beta) {
            ++beta_cutoffs_;
            return stand_pat;
        }
        alpha = std::max(alpha, stand_pat);

        // 安静局面只延伸吃子。无升变的中国象棋无需生成升变走法。
        moves.erase(
            std::remove_if(moves.begin(), moves.end(), [&](Move move) {
                return is_empty(position.piece_at(move.to));
            }),
            moves.end());
        if (moves.empty()) {
            return stand_pat;
        }
    }

    // 被将军时不能使用静态评分作为候选，必须搜索全部合法应将。
    move_ordering_.order_moves(position, moves, std::nullopt, ply,
                               previous_move_info(history));
    int best_score = checked ? -kSearchInfinity : stand_pat;
    for (Move move : moves) {
        push_search_move(history, move);
        ++nodes_;
        const int score = -quiescence(history, -beta, -alpha, ply + 1, qply + 1);
        pop_search_move(history);

        best_score = std::max(best_score, score);
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++beta_cutoffs_;
            break;
        }
    }
    return best_score;
}

std::optional<int> Searcher::adjudication_score(const GameHistory& history, int ply) {
    const HistoryVerdict verdict = adjudicate_history(history);
    switch (verdict) {
        case HistoryVerdict::Ongoing:
            return std::nullopt;
        case HistoryVerdict::DrawByThreefoldRepetition:
        case HistoryVerdict::DrawByNoCapture:
            return 0;
        case HistoryVerdict::RedWinsByBlackPerpetualCheck:
        case HistoryVerdict::RedWinsByBlackPerpetualChase:
            return history.position().side_to_move() == Color::Red
                       ? kMateScore - ply
                       : -kMateScore + ply;
        case HistoryVerdict::BlackWinsByRedPerpetualCheck:
        case HistoryVerdict::BlackWinsByRedPerpetualChase:
            return history.position().side_to_move() == Color::Black
                       ? kMateScore - ply
                       : -kMateScore + ply;
    }
    return std::nullopt;
}

std::uint64_t Searcher::transposition_key(const GameHistory& history) {
    return history.position().hash() ^
           std::rotl(history.rule_context_hash(), 23) ^
           0x7265706574697469ULL;
}

int Searcher::score_to_tt(int score, int ply) {
    if (score >= kMateThreshold) {
        return score + ply;
    }
    if (score <= -kMateThreshold) {
        return score - ply;
    }
    return score;
}

int Searcher::score_from_tt(int score, int ply) {
    if (score >= kMateThreshold) {
        return score - ply;
    }
    if (score <= -kMateThreshold) {
        return score + ply;
    }
    return score;
}

}  // namespace xiangqi
