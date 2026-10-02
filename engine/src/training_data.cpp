#include "xiangqi/training_data.hpp"

#include "xiangqi/cycle_adjudicator.hpp"
#include "xiangqi/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace xiangqi {
namespace {

constexpr int kInitialNonKingMaterial = 9600;

/// @brief 去除文本行首尾的 ASCII 空白。
std::string trim(std::string text) {
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

/// @brief 创建数据标注统一使用的 Alpha-Beta 搜索参数。
SearchLimits make_limits(int depth, int quiescence_depth) {
    return SearchLimits{
        .depth = depth,
        .algorithm = SearchAlgorithm::AlphaBeta,
        .quiescence_depth = quiescence_depth,
        .use_transposition_table = true,
    };
}

/// @brief 校验所有数据生成模式共享的数值参数。
void validate_common(int depth, int quiescence_depth,
                     int maximum_absolute_score, double weight) {
    if (depth <= 0) {
        throw std::invalid_argument("search depth must be positive");
    }
    if (quiescence_depth < 0) {
        throw std::invalid_argument("quiescence depth cannot be negative");
    }
    if (maximum_absolute_score <= 0 || maximum_absolute_score >= kMateThreshold) {
        throw std::invalid_argument(
            "maximum absolute score must be between 1 and the mate threshold");
    }
    if (!std::isfinite(weight) || weight <= 0.0) {
        throw std::invalid_argument("sample weight must be finite and positive");
    }
}

/// @brief 将胜方转换为某条样本当前行棋方视角的 {-1,0,+1} 结果。
int perspective_result(std::optional<Color> winner, Color perspective) {
    if (!winner) {
        return 0;
    }
    return *winner == perspective ? 1 : -1;
}

/// @brief 根据规则裁决返回胜方；和棋或尚未结束时返回空。
std::optional<Color> history_winner(HistoryVerdict verdict) {
    switch (verdict) {
        case HistoryVerdict::RedWinsByBlackPerpetualCheck:
        case HistoryVerdict::RedWinsByBlackPerpetualChase:
            return Color::Red;
        case HistoryVerdict::BlackWinsByRedPerpetualCheck:
        case HistoryVerdict::BlackWinsByRedPerpetualChase:
            return Color::Black;
        case HistoryVerdict::Ongoing:
        case HistoryVerdict::DrawByThreefoldRepetition:
        case HistoryVerdict::DrawByNoCapture:
            return std::nullopt;
    }
    return std::nullopt;
}

/// @brief 把一条训练样本写成不依赖第三方 JSON 库的稳定 JSONL。
void write_json_sample(std::ostream& output, std::string_view fen, int score,
                       double weight, int depth, std::optional<Move> best_move,
                       std::string_view phase, std::string_view source,
                       std::optional<std::size_t> game,
                       std::optional<std::size_t> ply,
                       std::optional<int> result) {
    output << "{\"fen\":\"" << fen << "\",\"score\":" << score
           << ",\"weight\":" << std::setprecision(12) << weight
           << ",\"depth\":" << depth;
    if (best_move) {
        output << ",\"best_move\":\"" << move_to_string(*best_move) << '"';
    }
    output << ",\"phase\":\"" << phase << "\",\"source\":\"" << source << '"';
    if (game) {
        output << ",\"game\":" << *game;
    }
    if (ply) {
        output << ",\"ply\":" << *ply;
    }
    if (result) {
        output << ",\"result\":" << *result;
    }
    output << "}\n";
    if (!output) {
        throw std::runtime_error("failed to write training data");
    }
}

}  // namespace

struct TrainingDataGenerator::PendingSample {
    std::string fen;
    int score{0};
    double weight{1.0};
    int depth{0};
    std::optional<Move> best_move;
    GamePhase phase{GamePhase::Opening};
    Color perspective{Color::Red};
    std::size_t game{0};
    std::size_t ply{0};
};

struct TrainingDataGenerator::ScoredMove {
    Move move{};
    int score{0};
};

std::string_view game_phase_name(GamePhase phase) {
    switch (phase) {
        case GamePhase::Opening: return "opening";
        case GamePhase::Middlegame: return "middlegame";
        case GamePhase::Endgame: return "endgame";
    }
    return "unknown";
}

GamePhase classify_game_phase(const Position& position) {
    int material = 0;
    for (int square = 0; square < kBoardSize; ++square) {
        const Piece piece = position.piece_at(square);
        if (!is_empty(piece)) {
            material += piece_base_value(piece_type(piece));
        }
    }
    if (material * 100 >= kInitialNonKingMaterial * 80) {
        return GamePhase::Opening;
    }
    if (material * 100 >= kInitialNonKingMaterial * 35) {
        return GamePhase::Middlegame;
    }
    return GamePhase::Endgame;
}

TrainingDataGenerator::TrainingDataGenerator(
    const Evaluator& evaluator, std::size_t transposition_table_mb)
    : play_searcher_(evaluator, transposition_table_mb),
      label_searcher_(evaluator, transposition_table_mb),
      random_(1) {}

TrainingDataStats TrainingDataGenerator::label_positions(
    std::istream& input, std::ostream& output,
    const LabelPositionsConfig& config) {
    validate_common(config.depth, config.quiescence_depth,
                    config.maximum_absolute_score, config.weight);
    seen_positions_.clear();
    TrainingDataStats stats;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string fen = trim(std::move(line));
        if (fen.empty() || fen.front() == '#') {
            continue;
        }
        ++stats.input_positions;

        Position position;
        try {
            position = Position::from_fen(fen);
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "invalid FEN at input line " + std::to_string(line_number) +
                ": " + error.what());
        }
        if (config.deduplicate && !seen_positions_.insert(position.hash()).second) {
            ++stats.duplicate_positions;
            continue;
        }

        GameHistory history(std::move(position));
        const SearchResult result = label_searcher_.search(
            history, make_limits(config.depth, config.quiescence_depth));
        if (!result.best_move) {
            ++stats.terminal_positions;
            continue;
        }
        if (std::abs(result.score) > config.maximum_absolute_score) {
            ++stats.extreme_scores;
            continue;
        }
        write_json_sample(
            output, history.position().to_fen(), result.score, config.weight,
            config.depth, result.best_move,
            game_phase_name(classify_game_phase(history.position())), "labeled-fen",
            std::nullopt, std::nullopt, std::nullopt);
        ++stats.samples_written;
    }
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("failed while reading FEN input");
    }
    return stats;
}

std::optional<Move> TrainingDataGenerator::choose_diverse_move(
    GameHistory& history, const SelfPlayConfig& config) {
    Position candidate_position = history.position();
    std::vector<Move> moves = candidate_position.generate_legal_moves();
    if (moves.empty()) {
        return std::nullopt;
    }
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());
    const int child_depth = std::max(0, config.play_depth - 1);
    for (Move move : moves) {
        if (!history.play(move)) {
            throw std::logic_error("legal move was rejected while scoring candidates");
        }
        const SearchResult reply = play_searcher_.search(
            history, make_limits(child_depth, config.quiescence_depth));
        scored.push_back(ScoredMove{.move = move, .score = -reply.score});
        if (!history.undo_last()) {
            throw std::logic_error("failed to restore a scored opening candidate");
        }
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [](const ScoredMove& left, const ScoredMove& right) {
                         return left.score > right.score;
                     });

    const int minimum_score = scored.front().score - config.random_score_margin;
    std::size_t eligible = 0;
    const std::size_t limit = std::min(config.random_top_k, scored.size());
    while (eligible < limit && scored[eligible].score >= minimum_score) {
        ++eligible;
    }
    std::uniform_int_distribution<std::size_t> distribution(0, eligible - 1);
    return scored[distribution(random_)].move;
}

TrainingDataStats TrainingDataGenerator::self_play(
    std::ostream& output, const SelfPlayConfig& config) {
    validate_common(config.play_depth, config.quiescence_depth,
                    config.maximum_absolute_score, config.weight);
    validate_common(config.label_depth, config.quiescence_depth,
                    config.maximum_absolute_score, config.weight);
    if (config.games == 0) {
        throw std::invalid_argument("game count must be positive");
    }
    if (config.maximum_plies == 0) {
        throw std::invalid_argument("maximum plies must be positive");
    }
    if (config.minimum_sample_gap == 0 ||
        config.minimum_sample_gap > config.maximum_sample_gap) {
        throw std::invalid_argument("sample gap range is invalid");
    }
    if (config.random_top_k == 0) {
        throw std::invalid_argument("random top-k must be positive");
    }
    if (config.random_score_margin < 0) {
        throw std::invalid_argument("random score margin cannot be negative");
    }

    random_.seed(config.seed);
    seen_positions_.clear();
    TrainingDataStats stats;
    std::uniform_int_distribution<std::size_t> gap_distribution(
        config.minimum_sample_gap, config.maximum_sample_gap);

    for (std::size_t game = 0; game < config.games; ++game) {
        GameHistory history;
        std::vector<PendingSample> samples;
        std::size_t next_sample = config.sample_start_ply;
        std::optional<Color> winner;
        bool result_known = false;

        while (history.ply_count() < config.maximum_plies) {
            const HistoryVerdict verdict = adjudicate_history(history);
            if (verdict != HistoryVerdict::Ongoing) {
                winner = history_winner(verdict);
                result_known = true;
                break;
            }

            Position legal_move_position = history.position();
            std::vector<Move> legal_moves = legal_move_position.generate_legal_moves();
            if (legal_moves.empty()) {
                winner = opposite(history.position().side_to_move());
                result_known = true;
                break;
            }

            const std::size_t ply = history.ply_count();
            const bool should_sample = ply >= next_sample;
            std::optional<SearchResult> label_result;
            if (should_sample) {
                label_result = label_searcher_.search(
                    history, make_limits(config.label_depth, config.quiescence_depth));
                next_sample += gap_distribution(random_);

                const std::uint64_t hash = history.position().hash();
                const bool unique = !config.deduplicate || seen_positions_.insert(hash).second;
                if (!unique) {
                    ++stats.duplicate_positions;
                } else if (!label_result->best_move) {
                    ++stats.terminal_positions;
                } else if (std::abs(label_result->score) >
                           config.maximum_absolute_score) {
                    ++stats.extreme_scores;
                } else {
                    samples.push_back(PendingSample{
                        .fen = history.position().to_fen(),
                        .score = label_result->score,
                        .weight = config.weight,
                        .depth = config.label_depth,
                        .best_move = label_result->best_move,
                        .phase = classify_game_phase(history.position()),
                        .perspective = history.position().side_to_move(),
                        .game = game,
                        .ply = ply,
                    });
                }
            }

            std::optional<Move> move;
            if (ply < config.random_opening_plies) {
                move = choose_diverse_move(history, config);
            } else if (label_result && config.play_depth == config.label_depth) {
                move = label_result->best_move;
            } else {
                const SearchResult play_result = play_searcher_.search(
                    history, make_limits(config.play_depth, config.quiescence_depth));
                move = play_result.best_move;
            }
            if (!move) {
                winner = opposite(history.position().side_to_move());
                result_known = true;
                break;
            }
            if (!history.play(*move)) {
                throw std::logic_error("searcher returned an illegal self-play move");
            }
        }

        for (const PendingSample& sample : samples) {
            write_json_sample(
                output, sample.fen, sample.score, sample.weight, sample.depth,
                sample.best_move, game_phase_name(sample.phase), "selfplay",
                sample.game, sample.ply,
                result_known
                    ? std::optional<int>(
                          perspective_result(winner, sample.perspective))
                    : std::nullopt);
            ++stats.samples_written;
        }
        ++stats.games;
    }
    return stats;
}

}  // namespace xiangqi
