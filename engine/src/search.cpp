#include "xiangqi/search.hpp"

#include "xiangqi/evaluation.hpp"

#include <algorithm>
#include <stdexcept>

namespace xiangqi {
namespace {

/// @brief 计算用于走法排序的吃子启发分。
/// @param position 走法执行前的局面。
/// @param move 待评分走法。
/// @return 吃子走法得到较高分；同为吃子时优先高价值目标和低价值攻击者。
int move_order_score(const Position& position, Move move) {
    const Piece captured = position.piece_at(move.to);
    if (is_empty(captured)) {
        return 0;
    }
    const Piece moving = position.piece_at(move.from);
    return 10000 + piece_base_value(piece_type(captured)) * 16 -
           piece_base_value(piece_type(moving));
}

/// @brief 将无合法走法的节点转换为当前行棋方的失败分数。
/// @param ply 当前节点距离根节点的半回合数。
/// @return 带将死距离的负分；越早失败，绝对值越大。
int terminal_loss_score(int ply) {
    return -kMateScore + ply;
}

}  // namespace

Searcher::Searcher(std::size_t transposition_table_mb)
    : transposition_table_(transposition_table_mb) {}

void Searcher::clear_transposition_table() {
    transposition_table_.clear();
    cached_quiescence_depth_.reset();
}

SearchResult Searcher::search(Position& position, const SearchLimits& limits) {
    if (limits.depth < 0) {
        throw std::invalid_argument("search depth cannot be negative");
    }
    if (limits.quiescence_depth < 0) {
        throw std::invalid_argument("quiescence depth cannot be negative");
    }

    nodes_ = 0;
    beta_cutoffs_ = 0;
    quiescence_nodes_ = 0;
    tt_hits_ = 0;
    tt_cutoffs_ = 0;
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
        result = search_iteration(position, 0, limits.algorithm, std::nullopt);
        result.iterations.push_back(IterationResult{
            .depth = 0,
            .best_move = result.best_move,
            .score = result.score,
            .nodes = nodes_ - nodes_before,
            .beta_cutoffs = beta_cutoffs_ - cutoffs_before,
            .quiescence_nodes = quiescence_nodes_ - qnodes_before,
            .tt_hits = tt_hits_ - tt_hits_before,
            .tt_cutoffs = tt_cutoffs_ - tt_cutoffs_before,
        });
        result.nodes = nodes_;
        result.beta_cutoffs = beta_cutoffs_;
        result.quiescence_nodes = quiescence_nodes_;
        result.tt_hits = tt_hits_;
        result.tt_cutoffs = tt_cutoffs_;
        return result;
    }

    std::optional<Move> previous_best;
    for (int current_depth = 1; current_depth <= limits.depth; ++current_depth) {
        const std::uint64_t nodes_before = nodes_;
        const std::uint64_t cutoffs_before = beta_cutoffs_;
        const std::uint64_t qnodes_before = quiescence_nodes_;
        const std::uint64_t tt_hits_before = tt_hits_;
        const std::uint64_t tt_cutoffs_before = tt_cutoffs_;

        SearchResult iteration = search_iteration(
            position, current_depth, limits.algorithm, previous_best);
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
    return result;
}

SearchResult Searcher::search_iteration(Position& position, int depth,
                                        SearchAlgorithm algorithm,
                                        std::optional<Move> previous_best) {
    ++nodes_;
    SearchResult result;
    result.depth = depth;

    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        result.score = terminal_loss_score(0);
        return result;
    }
    if (depth == 0) {
        result.score = quiescence(
            position, -kSearchInfinity, kSearchInfinity, 0, 0);
        return result;
    }

    std::optional<Move> preferred_move = previous_best;
    if (use_transposition_table_) {
        if (const TTEntry* entry = transposition_table_.probe(position.hash())) {
            ++tt_hits_;
            if (!preferred_move) {
                preferred_move = entry->best_move;
            }
        }
    }
    order_moves(position, moves, preferred_move);

    int best_score = -kSearchInfinity;
    Move best_move = moves.front();
    int alpha = -kSearchInfinity;
    constexpr int beta = kSearchInfinity;

    for (Move move : moves) {
        const UndoInfo undo = position.do_move(move);
        int score = 0;
        if (algorithm == SearchAlgorithm::Negamax) {
            score = -negamax(position, depth - 1, 1);
        } else {
            score = -alpha_beta(position, depth - 1, -beta, -alpha, 1);
        }
        position.undo_move(move, undo);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (algorithm == SearchAlgorithm::AlphaBeta) {
            alpha = std::max(alpha, score);
        }
    }

    result.best_move = best_move;
    result.score = best_score;
    if (use_transposition_table_) {
        transposition_table_.store(position.hash(), depth, score_to_tt(best_score, 0),
                                   TTBound::Exact, best_move);
    }
    return result;
}

int Searcher::negamax(Position& position, int depth, int ply) {
    ++nodes_;
    if (depth == 0) {
        return quiescence(position, -kSearchInfinity, kSearchInfinity, ply, 0);
    }
    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }

    order_moves(position, moves);
    int best_score = -kSearchInfinity;
    for (Move move : moves) {
        const UndoInfo undo = position.do_move(move);
        const int score = -negamax(position, depth - 1, ply + 1);
        position.undo_move(move, undo);
        best_score = std::max(best_score, score);
    }
    return best_score;
}

int Searcher::alpha_beta(Position& position, int depth, int alpha, int beta, int ply) {
    ++nodes_;
    if (depth == 0) {
        return quiescence(position, alpha, beta, ply, 0);
    }

    const int original_alpha = alpha;
    std::optional<Move> tt_move;
    if (use_transposition_table_) {
        if (const TTEntry* entry = transposition_table_.probe(position.hash())) {
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

    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }

    order_moves(position, moves, tt_move);
    int best_score = -kSearchInfinity;
    Move best_move = moves.front();
    for (Move move : moves) {
        const UndoInfo undo = position.do_move(move);
        const int score = -alpha_beta(position, depth - 1, -beta, -alpha, ply + 1);
        position.undo_move(move, undo);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++beta_cutoffs_;
            break;
        }
    }

    if (use_transposition_table_) {
        TTBound bound = TTBound::Exact;
        if (best_score <= original_alpha) {
            bound = TTBound::Upper;
        } else if (best_score >= beta) {
            bound = TTBound::Lower;
        }
        transposition_table_.store(position.hash(), depth,
                                   score_to_tt(best_score, ply), bound, best_move);
    }
    return best_score;
}

int Searcher::quiescence(Position& position, int alpha, int beta, int ply, int qply) {
    ++quiescence_nodes_;

    const bool checked = position.in_check(position.side_to_move());
    std::vector<Move> moves = position.generate_legal_moves();
    if (moves.empty()) {
        return terminal_loss_score(ply);
    }

    const int stand_pat = evaluate(position);
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
    order_moves(position, moves);
    int best_score = checked ? -kSearchInfinity : stand_pat;
    for (Move move : moves) {
        const UndoInfo undo = position.do_move(move);
        ++nodes_;
        const int score = -quiescence(position, -beta, -alpha, ply + 1, qply + 1);
        position.undo_move(move, undo);

        best_score = std::max(best_score, score);
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            ++beta_cutoffs_;
            break;
        }
    }
    return best_score;
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

void Searcher::order_moves(const Position& position, std::vector<Move>& moves,
                           std::optional<Move> preferred_move) {
    std::stable_sort(moves.begin(), moves.end(), [&](Move lhs, Move rhs) {
        return move_order_score(position, lhs) > move_order_score(position, rhs);
    });
    if (preferred_move) {
        const auto preferred = std::find(moves.begin(), moves.end(), *preferred_move);
        if (preferred != moves.end()) {
            std::iter_swap(moves.begin(), preferred);
        }
    }
}

}  // namespace xiangqi
