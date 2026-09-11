#include "xiangqi/cycle_adjudicator.hpp"
#include "xiangqi/evaluation.hpp"
#include "xiangqi/position.hpp"
#include "xiangqi/game_history.hpp"
#include "xiangqi/halfka.hpp"
#include "xiangqi/move_ordering.hpp"
#include "xiangqi/nnue.hpp"
#include "xiangqi/search.hpp"
#include "xiangqi/transposition_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {

using xiangqi::Color;
using xiangqi::Move;
using xiangqi::Position;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

Move move(const std::string& text) {
    const auto parsed = xiangqi::move_from_string(text);
    if (!parsed) {
        throw std::runtime_error("invalid move in test: " + text);
    }
    return *parsed;
}

bool contains(const std::vector<Move>& moves, const std::string& text) {
    return std::find(moves.begin(), moves.end(), move(text)) != moves.end();
}

bool nnue_accumulators_match_refresh(
    const xiangqi::NnueNetwork& network,
    const xiangqi::NnueEvaluationState& state,
    const Position& position) {
    for (Color perspective : {Color::Red, Color::Black}) {
        const xiangqi::NnueAccumulator refreshed =
            network.refresh_accumulator(position, perspective);
        const xiangqi::NnueAccumulator& incremental =
            state.accumulator(perspective);
        for (std::size_t neuron = 0;
             neuron < xiangqi::kNnueAccumulatorSize; ++neuron) {
            if (std::abs(refreshed[neuron] - incremental[neuron]) >= 0.0001F) {
                return false;
            }
        }
    }
    return true;
}

void test_initial_position() {
    Position position = Position::initial();
    expect(position.to_fen() ==
               "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
           "initial FEN round trip");
    expect(position.generate_legal_moves().size() == 44,
           "initial position has 44 legal moves");
    expect(xiangqi::perft(position, 2) == 1920,
           "initial position perft(2) is 1920");
    expect(xiangqi::perft(position, 3) == 79666,
           "initial position perft(3) is 79666");
}

void test_make_and_undo() {
    Position position = Position::initial();
    const Position original = position;
    const std::uint64_t original_hash = position.hash();
    const Move m = move("b0c2");
    expect(position.is_legal_move(m), "red horse b0c2 is legal initially");
    const auto undo = position.do_move(m);
    position.undo_move(m, undo);
    expect(position == original, "make/undo restores the complete position");
    expect(position.hash() == original_hash, "make/undo restores the Zobrist hash");
}

void test_horse_leg() {
    Position position = Position::from_fen("3k5/9/9/9/9/9/9/9/1P7/1N2K4 w");
    const auto moves = position.generate_legal_moves();
    expect(!contains(moves, "b0a2"), "blocked horse cannot move through its leg");
    expect(!contains(moves, "b0c2"), "one blocked horse leg blocks both matching jumps");
    expect(contains(moves, "b0d1"), "unblocked horse direction remains legal");
}

void test_cannon_screen() {
    Position position = Position::from_fen("4k4/4r4/9/4p4/9/9/4C4/9/9/3K5 w");
    const auto moves = position.generate_legal_moves();
    expect(contains(moves, "e3e8"), "cannon captures over exactly one screen");
    expect(!contains(moves, "e3e9"), "moves never capture the enemy king directly");
    expect(!contains(moves, "e3e6"), "cannon cannot move onto its screen");
}

void test_elephant_and_pawn_boundaries() {
    Position elephant = Position::from_fen("3k5/9/9/9/9/9/4B4/9/9/4K4 w");
    const auto elephant_moves = elephant.generate_legal_moves();
    expect(!contains(elephant_moves, "e3c5"), "red elephant cannot cross the river");

    Position blocked = Position::from_fen("3k5/9/9/9/9/9/4B4/3P5/9/4K4 w");
    expect(!contains(blocked.generate_legal_moves(), "e3c1"),
           "blocked elephant cannot cross its eye");

    Position before_river = Position::from_fen("3k5/9/9/9/9/9/4P4/9/9/4K4 w");
    const auto before_moves = before_river.generate_legal_moves();
    expect(contains(before_moves, "e3e4"), "red pawn moves forward");
    expect(!contains(before_moves, "e3d3"), "red pawn cannot move sideways before river");

    Position after_river = Position::from_fen("3k5/9/9/9/4P4/9/9/9/9/4K4 w");
    const auto after_moves = after_river.generate_legal_moves();
    expect(contains(after_moves, "e5d5") && contains(after_moves, "e5f5"),
           "red pawn can move sideways after river");
}

void test_flying_generals_and_pin() {
    Position facing = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w");
    expect(facing.in_check(Color::Red) && facing.in_check(Color::Black),
           "facing generals attack each other");
    expect(!facing.is_square_attacked(xiangqi::make_square(4, 4), Color::Black),
           "flying general does not attack an empty square");

    Position blocked = Position::from_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 w");
    expect(!blocked.in_check(Color::Red), "a blocking rook prevents flying generals");
    expect(!blocked.is_legal_move(move("e4d4")),
           "the blocker cannot expose flying generals");
    expect(blocked.is_legal_move(move("e4e5")),
           "the blocker may remain on the generals' file");
}

void test_check_evasion() {
    Position position = Position::from_fen("3k5/9/9/9/9/9/9/4r4/9/4K4 w");
    expect(position.in_check(Color::Red), "red is in rook check");
    const auto moves = position.generate_legal_moves();
    expect(!moves.empty(), "red has check evasions");
    for (Move m : moves) {
        const auto undo = position.do_move(m);
        expect(!position.in_check(Color::Red),
               "every generated response resolves the check: " + xiangqi::move_to_string(m));
        position.undo_move(m, undo);
    }
}

void test_invalid_fen() {
    bool threw = false;
    try {
        static_cast<void>(Position::from_fen("9/9/9/9/9/9/9/9/9/9 w"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "FEN without kings is rejected");
}

void test_halfka_features() {
    const Position initial = Position::initial();
    const xiangqi::HalfKAInput input = xiangqi::extract_halfka_input(initial);

    expect(xiangqi::kHalfKAFeatureDimensions == 11340,
           "HalfKA has nine king buckets, fourteen piece channels and ninety squares");
    expect(input[static_cast<std::size_t>(Color::Red)].size() == 32 &&
               input[static_cast<std::size_t>(Color::Black)].size() == 32,
           "HalfKA includes every piece, including both kings");
    expect(input[static_cast<std::size_t>(Color::Red)] ==
               input[static_cast<std::size_t>(Color::Black)],
           "rotationally symmetric initial position has equal relative-perspective features");

    for (const auto& features : input) {
        expect(std::all_of(features.begin(), features.end(), [](std::size_t index) {
                   return index < xiangqi::kHalfKAFeatureDimensions;
               }),
               "every HalfKA index lies inside the feature vector");
        auto sorted = features;
        std::sort(sorted.begin(), sorted.end());
        expect(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
               "each occupied square activates a distinct HalfKA feature");
    }

    const Position kings = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w");
    const auto king_features = xiangqi::extract_halfka_input(kings);
    expect(king_features[static_cast<std::size_t>(Color::Red)].size() == 2 &&
               king_features[static_cast<std::size_t>(Color::Black)].size() == 2,
           "HalfKA explicitly emits own-king and opponent-king features");
    expect(king_features[static_cast<std::size_t>(Color::Red)] ==
               king_features[static_cast<std::size_t>(Color::Black)],
           "both king-only perspectives use the same canonical orientation");
}

void test_game_history_and_repetition() {
    const Position initial = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/R8/4K4 w");
    xiangqi::GameHistory history(initial);
    const std::uint64_t initial_rule_context = history.rule_context_hash();

    expect(history.ply_count() == 0, "new history starts at ply zero");
    expect(history.position_hashes().size() == 1,
           "hash timeline starts with the initial position");
    expect(history.position_hashes().front() == initial.hash(),
           "first timeline hash belongs to the initial position");
    expect(history.no_capture_plies() == 0,
           "new history starts with zero no-capture plies");
    expect(history.current_position_occurrences() == 1,
           "initial position is recorded once");
    expect(!history.play(move("e0e3")),
           "history rejects an illegal move without changing state");
    expect(history.position() == initial, "illegal move leaves history unchanged");

    const std::string cycle[]{"a1a2", "e9d9", "a2a1", "d9e9"};
    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "history accepts legal cycle move " + text);
        }
    }

    expect(history.ply_count() == 8, "history stores every executed ply");
    expect(history.position_hashes().size() == history.ply_count() + 1,
           "hash timeline contains one position per ply plus the initial position");
    expect(history.no_capture_plies() == 8,
           "non-capturing moves increment the no-capture counter");
    expect(history.entries().front().hash_before == initial.hash() &&
               history.entries().front().hash_after == history.position_hashes()[1],
           "history entry records hashes on both sides of a move");
    expect(history.current_position_occurrences() == 3,
           "initial position plus two cycles is counted three times");
    expect(history.is_repetition(), "three occurrences trigger repetition indicator");
    expect(history.rule_context_hash() != initial_rule_context,
           "rule context distinguishes the same board reached through a longer history");

    for (int i = 0; i < 8; ++i) {
        expect(history.undo_last(), "history can undo every stored move");
    }
    expect(!history.undo_last(), "undo on empty history returns false");
    expect(history.position() == initial, "undoing all moves restores initial position");
    expect(history.position_hashes().size() == 1,
           "undo removes hashes belonging to future positions");
    expect(history.no_capture_plies() == 0,
           "undo restores the previous no-capture counter");
    expect(history.current_position_occurrences() == 1,
           "undo removes future positions from repetition history");
    expect(history.rule_context_hash() == initial_rule_context,
           "undo restores the original rule-context hash");
}

void test_history_check_and_capture_metadata() {
    xiangqi::GameHistory history(Position::from_fen(
        "3k5/9/9/9/4p4/9/9/9/R8/4K4 w"));

    expect(history.play(move("a1d1")), "rook checking move is legal");
    const auto checking_entry = history.last_entry();
    expect(checking_entry.has_value() && checking_entry->gave_check,
           "history records whether the move gave check");

    const Position capture_position = Position::from_fen(
        "3k5/9/9/9/4p4/9/9/p8/R8/4K4 w");
    xiangqi::GameHistory capture(capture_position, 12);
    expect(capture.no_capture_plies() == 12,
           "history accepts a no-capture counter when loading a position");
    expect(capture.play(move("a1a2")), "rook capture is legal");
    const auto capture_entry = capture.last_entry();
    expect(capture_entry.has_value() &&
               !xiangqi::is_empty(capture_entry->captured) &&
               xiangqi::piece_type(capture_entry->captured) == xiangqi::PieceType::Pawn,
           "history records the captured piece");
    expect(capture_entry.has_value() &&
               capture_entry->previous_no_capture_plies == 12,
           "history entry preserves the counter from before the move");
    expect(capture.no_capture_plies() == 0,
           "capture resets the no-capture counter");
    expect(capture.undo_last(), "capture can be undone");
    expect(capture.position() == capture_position &&
               capture.no_capture_plies() == 12,
           "undo restores both the board and no-capture counter");

    expect(capture.play(move("a1b1")), "rook non-capture is legal");
    expect(capture.no_capture_plies() == 13,
           "non-capture continues a counter loaded from an external position");

    capture.reset(capture_position, 7);
    expect(capture.ply_count() == 0 &&
               capture.position_hashes().size() == 1 &&
               capture.no_capture_plies() == 7,
           "reset clears moves and installs the supplied no-capture counter");
}

void test_cycle_extraction_and_adjudication() {
    const Position initial = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/R8/4K4 w");
    xiangqi::GameHistory history(initial);
    const std::string cycle[]{"a1a2", "e9d9", "a2a1", "d9e9"};

    for (const std::string& text : cycle) {
        expect(history.play(move(text)), "first cycle move is legal: " + text);
    }
    expect(!xiangqi::find_repetition_window(history).has_value(),
           "two occurrences are insufficient for the threefold rule");

    for (const std::string& text : cycle) {
        expect(history.play(move(text)), "second cycle move is legal: " + text);
    }

    const auto extracted = xiangqi::find_repetition_window(history);
    expect(extracted.has_value() &&
               extracted->occurrence_plies == std::vector<std::size_t>{0, 4, 8},
           "extractor returns the latest three matching position boundaries");

    const xiangqi::RepetitionAdjudication neutral =
        xiangqi::adjudicate_repetition(history);
    expect(neutral.verdict == xiangqi::RepetitionVerdict::Draw,
           "a threefold repetition without unilateral perpetual check is a draw");
    expect(xiangqi::adjudicate_history(history) ==
               xiangqi::HistoryVerdict::DrawByThreefoldRepetition,
           "history adjudicator reports a threefold-repetition draw");

    xiangqi::GameHistory different_paths(initial);
    for (const std::string& text : cycle) {
        expect(different_paths.play(move(text)),
               "first differing-path move is legal: " + text);
    }
    const std::string other_cycle[]{"a1b1", "e9f9", "b1a1", "f9e9"};
    for (const std::string& text : other_cycle) {
        expect(different_paths.play(move(text)),
               "second differing-path move is legal: " + text);
    }
    expect(different_paths.current_position_occurrences() == 3,
           "same boundary position can occur through different paths");
    const auto different_result = xiangqi::adjudicate_repetition(different_paths);
    expect(different_result.verdict == xiangqi::RepetitionVerdict::Draw &&
               different_result.window.has_value() &&
               different_result.window->occurrence_plies ==
                   std::vector<std::size_t>{0, 4, 8},
           "three equal positions are adjudicated even when return paths differ");
}

void test_perpetual_check_adjudication() {
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/3R5/9/9/4p4/9/9/9/9/4K4 w"));
    const std::string cycle[]{
        "d8e8", "e9f9", "e8f8", "f9e9",
        "f8e8", "e9d9", "e8d8", "d9e9",
    };

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "perpetual-check move is legal: " + text);
        }
    }

    const xiangqi::RepetitionAdjudication result =
        xiangqi::adjudicate_repetition(history);
    expect(result.window.has_value() &&
               result.window->occurrence_plies ==
                   std::vector<std::size_t>{0, 8, 16},
           "perpetual-check repetition contains three matching boundaries");
    expect(result.verdict == xiangqi::RepetitionVerdict::RedPerpetualCheck,
           "red is identified as the sole perpetual-checking side");
    expect(xiangqi::adjudicate_history(history) ==
               xiangqi::HistoryVerdict::BlackWinsByRedPerpetualCheck,
           "red perpetual check is converted into a black win");

    expect(history.undo_last(), "last perpetual-check response can be undone");
    expect(xiangqi::adjudicate_repetition(history).verdict ==
               xiangqi::RepetitionVerdict::None,
           "breaking the repeated boundary removes the cycle verdict");
}

void test_perpetual_chase_adjudication() {
    // 两个红兵分别作为炮架。红炮在 a1、b1 间往返，每一步都新捉同一辆黑车；
    // 黑车在 a4、b4 间躲避并最终回到原局面。
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/4p4/r8/9/PP7/1C7/4K4 w"));
    const std::string cycle[]{"b1a1", "a4b4", "a1b1", "b4a4"};

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "perpetual-chase move is legal: " + text);
        }
    }

    const xiangqi::RepetitionAdjudication result =
        xiangqi::adjudicate_repetition(history);
    expect(result.verdict == xiangqi::RepetitionVerdict::RedPerpetualChase,
           "red cannon is identified as perpetually chasing the same black rook");
    expect(xiangqi::adjudicate_history(history) ==
               xiangqi::HistoryVerdict::BlackWinsByRedPerpetualChase,
           "red perpetual chase is converted into a black win");
}

void test_unprotected_piece_perpetual_chase() {
    // 红炮在 a1、c1 间借炮架持续追同一匹无根黑马，覆盖普通真捉分支，
    // 而不是“炮捉车”弱子捉强子的直接认定分支。
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/4p4/n8/9/P1P6/2C6/4K4 w"));
    const std::string cycle[]{"c1a1", "a4c3", "a1c1", "c3a4"};

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "unprotected-chase move is legal: " + text);
        }
    }

    expect(xiangqi::adjudicate_repetition(history).verdict ==
               xiangqi::RepetitionVerdict::RedPerpetualChase,
           "repeatedly attacking the same unprotected horse is a perpetual chase");
}

void test_pawn_perpetual_chase_is_allowed() {
    // 过河红兵横向往返并持续攻击同一辆黑车。a5 红车保护兵，避免黑车的反向
    // 攻击被误认为另一方长捉。按采用的简化规则，兵卒长捉不负。
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/4r4/R2P5/9/4P4/9/9/4K4 w"));
    const std::string cycle[]{"d5e5", "e6d6", "e5d5", "d6e6"};

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "pawn-chase cycle move is legal: " + text);
        }
    }

    expect(xiangqi::adjudicate_repetition(history).verdict ==
               xiangqi::RepetitionVerdict::Draw,
           "a pawn repeatedly chasing the same piece remains a repetition draw");
}

void test_king_perpetual_chase_is_allowed() {
    // 红帅在 d0、e0 间往返并持续贴捉同一门黑炮。炮与帅相邻且没有炮架，
    // 不会反过来将军；将帅作为追逐者时不承担长捉责任。
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/3c5/4K4 w"));
    const std::string cycle[]{"e0d0", "d1e1", "d0e0", "e1d1"};

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "king-chase cycle move is legal: " + text);
        }
    }

    expect(xiangqi::adjudicate_repetition(history).verdict ==
               xiangqi::RepetitionVerdict::Draw,
           "a king repeatedly chasing the same piece remains a repetition draw");
}

void test_protected_piece_is_not_perpetually_chased() {
    // 红炮在 a1、c1 间借炮架追黑马；a5、c4 的黑车分别保护马在 a4、c3
    // 的位置。炮若吃马会被合法反吃，因此每一步都不属于真捉。
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/r3p4/n1r6/9/P1P6/2C6/4K4 w"));
    const std::string cycle[]{"c1a1", "a4c3", "a1c1", "c3a4"};

    for (int repetition = 0; repetition < 2; ++repetition) {
        for (const std::string& text : cycle) {
            expect(history.play(move(text)), "protected-chase cycle move is legal: " + text);
        }
    }

    expect(xiangqi::adjudicate_repetition(history).verdict ==
               xiangqi::RepetitionVerdict::Draw,
           "repeated attacks on a legally protected horse are not a perpetual chase");
}

void test_no_capture_draw_adjudication() {
    const Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/R8/4K4 w");
    xiangqi::GameHistory history(position, xiangqi::kNoCaptureDrawPlies - 1);

    expect(xiangqi::adjudicate_history(history) == xiangqi::HistoryVerdict::Ongoing,
           "119 no-capture plies do not trigger the 60-move draw");

    expect(history.play(move("a1a2")), "final non-capture before limit is legal");
    expect(xiangqi::adjudicate_history(history) ==
               xiangqi::HistoryVerdict::DrawByNoCapture,
           "120 no-capture plies trigger the 60-move draw");
    expect(history.undo_last() && history.no_capture_plies() == 119 &&
               xiangqi::adjudicate_history(history) == xiangqi::HistoryVerdict::Ongoing,
           "undo restores the no-capture counter below the draw threshold");

    history.reset(Position::from_fen(
        "3k5/9/9/9/4p4/9/9/p8/R8/4K4 w"),
        xiangqi::kNoCaptureDrawPlies - 1);
    expect(history.play(move("a1a2")), "capture before limit is legal");
    expect(history.no_capture_plies() == 0 &&
               xiangqi::adjudicate_history(history) == xiangqi::HistoryVerdict::Ongoing,
           "a capture resets the 60-move draw counter");
}

void test_evaluation_symmetry_and_material() {
    Position initial = Position::initial();
    expect(xiangqi::evaluate_breakdown(initial).total() == 0,
           "symmetric initial position evaluates to zero");
    expect(xiangqi::evaluate(initial) == 0,
           "initial position is equal from side-to-move perspective");

    // 移除黑方左车后，红方应当获得接近一车的明显优势。
    initial.set_piece(xiangqi::make_square(0, 0), xiangqi::kEmpty);
    const int red_score = xiangqi::evaluate_for(initial, Color::Red);
    const int black_score = xiangqi::evaluate_for(initial, Color::Black);
    expect(red_score > 850, "missing black rook gives red a large positive score");
    expect(black_score == -red_score,
           "evaluation changes sign when perspective changes");

    initial.set_side_to_move(Color::Black);
    expect(xiangqi::evaluate(initial) == black_score,
           "default evaluation follows side-to-move perspective");
}

void test_evaluator_interface_and_search_injection() {
    class ConstantEvaluator final : public xiangqi::Evaluator {
    public:
        [[nodiscard]] std::string_view name() const noexcept override {
            return "constant-test";
        }

        [[nodiscard]] int evaluate_for(
            const Position&, Color perspective) const override {
            ++calls;
            return perspective == Color::Red ? 137 : -137;
        }

        mutable int calls{0};
    } evaluator;

    const Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/9/4K4 w");
    expect(evaluator.name() == "constant-test",
           "Evaluator exposes its implementation name");
    expect(evaluator.evaluate(position) == 137,
           "Evaluator converts its perspective API to side-to-move scoring");

    Position search_position = position;
    xiangqi::Searcher searcher(evaluator);
    const xiangqi::SearchResult result = searcher.search(
        search_position,
        xiangqi::SearchLimits{.depth = 0,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0});
    expect(result.score == 137 && evaluator.calls >= 2,
           "Searcher evaluates leaf nodes through the injected Evaluator");
    expect(search_position == position,
           "custom evaluator search leaves the input position unchanged");
}

void test_nnue_accumulator_and_forward_inference() {
    const Position position = Position::from_fen(
        "4k4/9/9/9/9/9/9/9/9/4K4 w");
    xiangqi::NnueNetwork network;

    network.feature_biases()[0] = 0.1F;
    const auto features = xiangqi::active_halfka_features(position, Color::Red);
    for (std::size_t feature : features) {
        network.feature_weights()[
            feature * xiangqi::kNnueAccumulatorSize] = 0.2F;
    }

    const xiangqi::NnueAccumulator accumulator =
        network.refresh_accumulator(position, Color::Red);
    expect(std::abs(accumulator[0] - 0.5F) < 0.0001F,
           "NNUE accumulator adds its bias and every active HalfKA feature row");
    expect(accumulator[1] == 0.0F,
           "unconnected NNUE accumulator neurons remain at their bias");

    network.hidden_weights()[0] = 1.0F;
    network.output_weights()[0] = 100.0F;
    network.output_bias() = 1.4F;
    expect(std::abs(network.forward(position, Color::Red) - 51.4F) < 0.0001F,
           "NNUE forward pass applies concatenation, hidden layer and output layer");
    expect(network.evaluate(position, Color::Red) == 51,
           "NNUE evaluation rounds the float output to an engine score");

    network.feature_biases()[0] = 10.0F;
    expect(std::abs(network.forward(position, Color::Red) - 101.4F) < 0.0001F,
           "NNUE feature-transform activation is clipped to one");
}

void test_nnue_incremental_accumulator_updates() {
    xiangqi::NnueNetwork network;
    for (std::size_t feature = 0;
         feature < xiangqi::kHalfKAFeatureDimensions; ++feature) {
        for (std::size_t neuron = 0; neuron < 4; ++neuron) {
            const int pattern = static_cast<int>((feature + neuron * 7U) % 23U) - 11;
            network.feature_weights()[
                feature * xiangqi::kNnueAccumulatorSize + neuron] =
                static_cast<float>(pattern) * 0.003F;
        }
    }

    Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/4R4/9/4K4 w");
    xiangqi::NnueEvaluationState state(network, position);

    const auto exercise_move = [&](const std::string& text,
                                   const std::string& description) {
        const Move candidate = move(text);
        const Position before = position;
        const xiangqi::NnueAccumulator red_before = state.accumulator(Color::Red);
        const xiangqi::NnueAccumulator black_before = state.accumulator(Color::Black);
        const xiangqi::Piece moved = position.piece_at(candidate.from);
        const xiangqi::Piece captured = position.piece_at(candidate.to);
        const xiangqi::UndoInfo undo = position.do_move(candidate);
        state.push_move(position, candidate, moved, captured);

        expect(nnue_accumulators_match_refresh(network, state, position),
               description + " incremental update matches refresh");
        expect(state.stack_size() == 1,
               description + " stores one accumulator undo snapshot");

        state.pop_move();
        position.undo_move(candidate, undo);
        expect(position == before &&
                   state.accumulator(Color::Red) == red_before &&
                   state.accumulator(Color::Black) == black_before &&
                   state.stack_size() == 0,
               description + " pop restores the exact root state");
        expect(state.evaluate_for(position, Color::Red) ==
                   network.evaluate(position, Color::Red),
               description + " restored state remains synchronized");
    };

    exercise_move("e2d2", "quiet rook move");
    exercise_move("e2e5", "rook capture");
    exercise_move("e0d0", "king move with own-perspective refresh");

    struct PlayedMove {
        Move move{};
        xiangqi::UndoInfo undo{};
    };
    Position line = Position::initial();
    const Position line_root = line;
    xiangqi::NnueEvaluationState line_state(network, line);
    std::vector<PlayedMove> played;
    for (std::size_t ply = 0; ply < 24; ++ply) {
        std::vector<Move> moves = line.generate_legal_moves();
        if (moves.empty()) {
            break;
        }
        const Move candidate = moves[(ply * 17U + 3U) % moves.size()];
        const xiangqi::Piece moved = line.piece_at(candidate.from);
        const xiangqi::Piece captured = line.piece_at(candidate.to);
        const xiangqi::UndoInfo undo = line.do_move(candidate);
        line_state.push_move(line, candidate, moved, captured);
        played.push_back(PlayedMove{candidate, undo});
        expect(nnue_accumulators_match_refresh(network, line_state, line),
               "multi-ply incremental accumulators match a full refresh");
    }
    while (!played.empty()) {
        const PlayedMove last = played.back();
        played.pop_back();
        line_state.pop_move();
        line.undo_move(last.move, last.undo);
        expect(nnue_accumulators_match_refresh(network, line_state, line),
               "multi-ply accumulator undo matches a full refresh");
    }
    expect(line == line_root && line_state.stack_size() == 0,
           "multi-ply accumulator stack returns exactly to its root");

    const Position search_root = position;
    xiangqi::NnueEvaluator evaluator(std::move(network));
    xiangqi::Searcher searcher(evaluator);
    const xiangqi::SearchResult result = searcher.search(
        position,
        xiangqi::SearchLimits{.depth = 2,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0,
                              .use_transposition_table = false});
    expect(result.best_move.has_value(),
           "search can traverse multiple plies with incremental NNUE state");
    expect(position == search_root,
           "incremental NNUE search restores its root position");
}

void test_search_maintains_evaluation_state() {
    struct Counters {
        int pushes{0};
        int pops{0};
        int evaluations{0};
    };

    class TrackingState final : public xiangqi::EvaluationState {
    public:
        explicit TrackingState(Counters& counters) : counters_(counters) {}

        void push_move(const Position&, Move, xiangqi::Piece,
                       xiangqi::Piece) override {
            ++counters_.pushes;
        }

        void pop_move() override {
            ++counters_.pops;
        }

        [[nodiscard]] int evaluate_for(
            const Position&, Color) const override {
            ++counters_.evaluations;
            return 0;
        }

    private:
        Counters& counters_;
    };

    class TrackingEvaluator final : public xiangqi::Evaluator {
    public:
        explicit TrackingEvaluator(Counters& counters) : counters_(counters) {}

        [[nodiscard]] std::string_view name() const noexcept override {
            return "tracking-test";
        }

        [[nodiscard]] int evaluate_for(const Position&, Color) const override {
            return 0;
        }

        [[nodiscard]] std::unique_ptr<xiangqi::EvaluationState> create_state(
            const Position&) const override {
            return std::make_unique<TrackingState>(counters_);
        }

    private:
        Counters& counters_;
    };

    Counters counters;
    TrackingEvaluator evaluator(counters);
    Position position = Position::initial();
    const Position original = position;
    xiangqi::Searcher searcher(evaluator);
    static_cast<void>(searcher.search(
        position,
        xiangqi::SearchLimits{.depth = 2,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0,
                              .use_transposition_table = false}));

    expect(counters.pushes > 0 && counters.pushes == counters.pops,
           "Searcher balances every evaluation-state push with a pop");
    expect(counters.evaluations > 0,
           "Searcher evaluates leaves through its search-specific state");
    expect(position == original,
           "evaluation-state maintenance preserves the root position");
}

void test_nnue_file_round_trip_and_search() {
    xiangqi::NnueNetwork network;
    network.feature_biases()[3] = 0.75F;
    network.feature_weights()[17] = -0.125F;
    network.hidden_biases()[2] = 0.5F;
    network.hidden_weights()[29] = 0.25F;
    network.output_weights()[2] = -3.5F;
    network.output_bias() = 73.0F;

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "chinese_cheese_ai_nnue_round_trip.bin";
    network.save(path);
    xiangqi::NnueNetwork loaded = xiangqi::NnueNetwork::load(path);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);

    expect(loaded.feature_biases()[3] == 0.75F &&
               loaded.feature_weights()[17] == -0.125F &&
               loaded.hidden_biases()[2] == 0.5F &&
               loaded.hidden_weights()[29] == 0.25F &&
               loaded.output_weights()[2] == -3.5F &&
               loaded.output_bias() == 73.0F,
           "NNUE binary file round trip preserves all layer parameter types");
    expect(!remove_error, "NNUE round-trip test removes its temporary model file");

    xiangqi::NnueNetwork constant_network;
    constant_network.output_bias() = 73.0F;
    xiangqi::NnueEvaluator evaluator(std::move(constant_network));
    expect(evaluator.name() == "nnue-halfka-f32",
           "NnueEvaluator exposes its network type");

    Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/9/4K4 w");
    const Position original = position;
    xiangqi::Searcher searcher(evaluator);
    const xiangqi::SearchResult result = searcher.search(
        position,
        xiangqi::SearchLimits{.depth = 0,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0});
    expect(result.score == 73,
           "Searcher obtains leaf scores from an injected NnueEvaluator");
    expect(position == original,
           "NNUE search leaves the input position unchanged");
}

void test_advanced_pawn_value() {
    Position home_pawn = Position::from_fen(
        "3k5/9/9/9/9/9/4P4/9/9/4K4 w");
    Position advanced_pawn = Position::from_fen(
        "3k5/9/9/9/4P4/9/9/9/9/4K4 w");

    expect(xiangqi::evaluate_for(advanced_pawn, Color::Red) >
               xiangqi::evaluate_for(home_pawn, Color::Red),
           "an advanced crossed-river pawn is worth more than a home-side pawn");
}

void test_piece_base_values() {
    expect(xiangqi::piece_base_value(xiangqi::PieceType::Rook) == 900,
           "rook base value is 900");
    expect(xiangqi::piece_base_value(xiangqi::PieceType::Cannon) == 450,
           "cannon base value is 450");
    expect(xiangqi::piece_base_value(xiangqi::PieceType::Horse) == 400,
           "horse base value is 400");
    expect(xiangqi::piece_base_value(xiangqi::PieceType::King) == 0,
           "king is excluded from static material scoring");
}

void test_capture_move_ordering() {
    Position position = Position::from_fen(
        "4k4/9/9/9/rp7/PRCPp4/9/9/9/4K4 w");
    const Move pawn_takes_rook = move("a4a5");
    const Move rook_takes_pawn = move("b4b5");
    const Move cannon_takes_pawn = move("c4e4");
    const Move quiet_move = move("c4c5");

    expect(xiangqi::capture_move_order_score(position, pawn_takes_rook) >
               xiangqi::capture_move_order_score(position, rook_takes_pawn),
           "capturing a rook is ordered before capturing a pawn");
    expect(xiangqi::capture_move_order_score(position, cannon_takes_pawn) >
               xiangqi::capture_move_order_score(position, rook_takes_pawn),
           "a lower-value attacker is preferred when capturing the same victim type");
    expect(xiangqi::capture_move_order_score(position, rook_takes_pawn) >
               xiangqi::capture_move_order_score(position, quiet_move),
           "captures are ordered before quiet moves");

    std::vector<Move> moves{
        quiet_move,
        rook_takes_pawn,
        pawn_takes_rook,
        cannon_takes_pawn,
    };
    xiangqi::MoveOrdering ordering;
    expect(!ordering.order_moves(position, moves, std::nullopt, 0, std::nullopt),
           "capture ordering reports no preferred move when none is supplied");
    expect(moves[0] == pawn_takes_rook && moves[1] == cannon_takes_pawn &&
               moves[2] == quiet_move && moves[3] == rook_takes_pawn,
           "SEE keeps profitable captures first and moves a losing capture behind quiet moves");

    expect(ordering.order_moves(position, moves, quiet_move, 0, std::nullopt) &&
               moves[0] == quiet_move,
           "a legal preferred move remains higher priority than every capture");
}

void test_static_exchange_evaluation() {
    Position poisoned = Position::from_fen(
        "3k5/9/9/9/4p4/9/r8/p8/R8/4K4 w");
    const Position poisoned_before = poisoned;
    expect(xiangqi::MoveOrdering::static_exchange_evaluation(
               poisoned, move("a1a2")) == -800,
           "SEE detects a rook losing eight hundred points after taking a protected pawn");
    expect(poisoned == poisoned_before,
           "SEE restores the complete position after a recapture sequence");

    Position winning = Position::from_fen(
        "3k5/9/9/9/4p4/9/9/r8/P8/4K4 w");
    expect(xiangqi::MoveOrdering::static_exchange_evaluation(
               winning, move("a1a2")) == 900,
           "SEE values an undefended rook capture using evaluator material values");
}

void test_quiet_check_ordering() {
    Position position = Position::from_fen(
        "3k5/9/9/9/9/9/9/9/R8/4K4 w");
    const Move quiet = move("a1a2");
    const Move checking = move("a1d1");
    const Position original = position;

    expect(xiangqi::MoveOrdering::gives_check(position, checking),
           "move ordering recognizes a non-capturing rook check");
    expect(!xiangqi::MoveOrdering::gives_check(position, quiet),
           "move ordering does not mark an ordinary rook move as check");

    xiangqi::MoveOrdering ordering;
    std::vector<Move> moves{quiet, checking};
    ordering.order_moves(position, moves, std::nullopt, 0, std::nullopt);
    expect(moves.front() == checking,
           "a safe quiet check is ordered before an ordinary quiet move");
    expect(position == original,
           "check detection and ordering restore the complete position");
}

void test_killer_counter_and_history_ordering() {
    Position position = Position::initial();
    const Move first = move("b0c2");
    const Move second = move("h0g2");
    const Move ordinary = move("a0a1");
    xiangqi::MoveOrdering killers;

    killers.record_quiet_beta_cutoff(position, first, 3, 4,
                                     std::nullopt, {});
    std::vector<Move> moves{ordinary, second, first};
    killers.order_moves(position, moves, std::nullopt, 3, std::nullopt);
    expect(moves.front() == first,
           "the first killer is preferred over ordinary quiet moves at the same ply");

    killers.record_quiet_beta_cutoff(position, second, 3, 4,
                                     std::nullopt, {});
    moves = {ordinary, first, second};
    killers.order_moves(position, moves, std::nullopt, 3, std::nullopt);
    expect(moves[0] == second && moves[1] == first,
           "a new first killer moves the previous first killer into the second slot");

    xiangqi::MoveOrdering contextual;
    const xiangqi::PreviousMoveInfo previous{
        move("a6a5"),
        xiangqi::make_piece(Color::Black, xiangqi::PieceType::Pawn),
    };
    contextual.record_quiet_beta_cutoff(position, second, 10, 64,
                                        std::nullopt, {});
    contextual.record_quiet_beta_cutoff(position, first, 11, 1,
                                        previous, {});

    moves = {ordinary, first, second};
    contextual.order_moves(position, moves, std::nullopt, 20, std::nullopt);
    expect(moves.front() == second,
           "history orders a repeatedly successful quiet move before low-history moves");

    moves = {ordinary, first, second};
    contextual.order_moves(position, moves, std::nullopt, 20, previous);
    expect(moves.front() == first,
           "Counter Move outranks quiet-move history when the previous move matches");
}

void test_piece_to_history_ordering() {
    Position learned_position = Position::from_fen(
        "3k5/9/9/9/9/9/9/9/R8/4K4 w");
    xiangqi::MoveOrdering ordering;
    ordering.record_quiet_beta_cutoff(
        learned_position, move("a1a2"), 3, 32, std::nullopt, {});

    // Main History 没见过 a3a2，但 Piece-to History 已学习过“红车走到 a2”。
    Position new_position = Position::from_fen(
        "3k5/9/9/9/9/9/R8/2C6/9/4K4 w");
    const Move cannon_to_a2 = move("c2a2");
    const Move rook_to_a2 = move("a3a2");
    std::vector<Move> moves{cannon_to_a2, rook_to_a2};
    ordering.order_moves(new_position, moves, std::nullopt, 20, std::nullopt);

    expect(moves.front() == rook_to_a2,
           "Piece-to History transfers a rook-to-target success across different origins");
}

void test_search_finds_obvious_capture() {
    Position position = Position::from_fen(
        "3k5/9/9/9/4p4/9/9/r8/R8/4K4 w");
    const Position original = position;
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult result = searcher.search(
        position,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});

    expect(result.best_move.has_value() && *result.best_move == move("a1a2"),
           "depth-one search captures an undefended enemy rook");
    expect(result.score > 0, "winning a rook produces a positive search score");
    expect(position == original, "search restores the root position");
}

void test_negamax_matches_alpha_beta() {
    Position negamax_position = Position::initial();
    Position alpha_beta_position = Position::initial();
    xiangqi::Searcher searcher;

    const xiangqi::SearchResult negamax = searcher.search(
        negamax_position,
        xiangqi::SearchLimits{.depth = 3,
                              .algorithm = xiangqi::SearchAlgorithm::Negamax});
    const xiangqi::SearchResult alpha_beta = searcher.search(
        alpha_beta_position,
        xiangqi::SearchLimits{.depth = 3,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});

    expect(negamax.score == alpha_beta.score,
           "Negamax and Alpha-Beta return the same depth-three score");
    expect(negamax.best_move == alpha_beta.best_move,
           "Negamax and Alpha-Beta choose the same depth-three move");
    expect(alpha_beta.nodes < negamax.nodes,
           "Alpha-Beta visits fewer nodes than plain Negamax");
    expect(alpha_beta.beta_cutoffs > 0,
           "Alpha-Beta reports at least one cutoff");
    expect(alpha_beta.pvs_researches > 0,
           "PVS performs a full-window re-search when a scout search raises alpha");
    expect(negamax.pvs_researches == 0,
           "plain Negamax does not perform PVS re-searches");
}

void test_search_terminal_and_depth_zero() {
    Position checkmate = Position::from_fen(
        "4k4/3PRP3/4P4/9/9/9/9/9/9/4K4 b");
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult mate_result = searcher.search(
        checkmate,
        xiangqi::SearchLimits{.depth = 3,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});
    expect(!mate_result.best_move.has_value(), "terminal position has no best move");
    expect(mate_result.score == -xiangqi::kMateScore,
           "side with no legal move receives a terminal loss score");

    Position initial = Position::initial();
    const xiangqi::SearchResult static_result = searcher.search(
        initial,
        xiangqi::SearchLimits{.depth = 0,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0});
    expect(!static_result.best_move.has_value(), "depth-zero search has no best move");
    expect(static_result.score == xiangqi::evaluate(initial),
           "depth-zero search returns the static evaluation");
}

void test_quiescence_avoids_poisoned_capture() {
    // 红车可以吃 a2 的卒，但随后会被 a3 的黑车吃回。
    Position without_qsearch = Position::from_fen(
        "3k5/9/9/9/4p4/9/r8/p8/R8/4K4 w");
    Position with_qsearch = without_qsearch;
    xiangqi::Searcher searcher;

    const xiangqi::SearchResult shallow = searcher.search(
        without_qsearch,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 0});
    const xiangqi::SearchResult quiet = searcher.search(
        with_qsearch,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 8});

    expect(shallow.best_move.has_value() && *shallow.best_move == move("a1a2"),
           "plain depth-one search takes the apparently free pawn");
    expect(quiet.best_move.has_value() && *quiet.best_move != move("a1a2"),
           "quiescence sees the rook recapture and avoids the poisoned pawn");
    expect(quiet.quiescence_nodes > 0, "quiescence node count is reported");
    expect(without_qsearch == with_qsearch,
           "both searches restore the complete root position");
}

void test_quiescence_searches_check_evasions() {
    Position checked = Position::from_fen(
        "3k5/9/9/9/4p4/9/9/4r4/9/4K4 w");
    const Position original = checked;
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult result = searcher.search(
        checked,
        xiangqi::SearchLimits{.depth = 0,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 8});

    expect(result.quiescence_nodes > 1,
           "quiescence expands legal evasions when the side to move is checked");
    expect(checked == original, "check-evasion quiescence restores the position");
}

void test_iterative_deepening_results() {
    Position position = Position::initial();
    const Position original = position;
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult result = searcher.search(
        position,
        xiangqi::SearchLimits{.depth = 3,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
                              .quiescence_depth = 8});

    expect(result.iterations.size() == 3,
           "iterative deepening completes depths one through three");
    for (std::size_t index = 0; index < result.iterations.size(); ++index) {
        expect(result.iterations[index].depth == static_cast<int>(index + 1),
               "iteration depths are strictly increasing");
        expect(result.iterations[index].best_move.has_value(),
               "every completed non-terminal iteration has a best move");
        expect(result.iterations[index].nodes > 0,
               "every completed iteration reports its own node count");
    }
    const xiangqi::IterationResult& last = result.iterations.back();
    expect(result.depth == last.depth && result.score == last.score &&
               result.best_move == last.best_move,
           "final search result equals the deepest completed iteration");
    expect(position == original,
           "all iterative-deepening passes restore the root position");
    std::uint64_t iteration_researches = 0;
    for (const xiangqi::IterationResult& iteration : result.iterations) {
        iteration_researches += iteration.pvs_researches;
    }
    expect(iteration_researches == result.pvs_researches,
           "per-iteration PVS re-search counts add up to the total");
}

void test_transposition_table_storage_and_replacement() {
    xiangqi::TranspositionTable table(1);
    table.new_search();
    const std::uint64_t first_key = 17;
    const std::uint64_t colliding_key = first_key + table.entry_count();

    table.store(first_key, 6, 123, xiangqi::TTBound::Exact, move("a0a1"));
    const xiangqi::TTEntry* first = table.probe(first_key);
    expect(first != nullptr && first->depth == 6 && first->score == 123 &&
               first->best_move == move("a0a1"),
           "transposition table returns a fully matching stored entry");

    table.store(first_key, 2, 50, xiangqi::TTBound::Upper, move("a0a2"));
    first = table.probe(first_key);
    expect(first != nullptr && first->depth == 6,
           "shallower result does not replace a deeper entry for the same key");

    table.store(colliding_key, 2, 77, xiangqi::TTBound::Lower, move("b0c2"));
    expect(table.probe(first_key) != nullptr && table.probe(colliding_key) == nullptr,
           "same-generation shallow collision does not evict a deeper entry");

    table.new_search();
    table.store(colliding_key, 2, 77, xiangqi::TTBound::Lower, move("b0c2"));
    expect(table.probe(first_key) == nullptr && table.probe(colliding_key) != nullptr,
           "a new generation may replace an old colliding entry");

    table.clear();
    expect(table.probe(colliding_key) == nullptr,
           "clearing the transposition table invalidates stored entries");
}

void test_search_uses_transposition_table() {
    Position first_position = Position::initial();
    Position second_position = Position::initial();
    Position uncached_position = Position::initial();
    xiangqi::Searcher cached_searcher(16);
    xiangqi::Searcher uncached_searcher(16);
    const xiangqi::SearchLimits cached_limits{
        .depth = 3,
        .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
        .quiescence_depth = 8,
        .use_transposition_table = true,
    };
    const xiangqi::SearchLimits uncached_limits{
        .depth = 3,
        .algorithm = xiangqi::SearchAlgorithm::AlphaBeta,
        .quiescence_depth = 8,
        .use_transposition_table = false,
    };

    const xiangqi::SearchResult first = cached_searcher.search(first_position, cached_limits);
    const xiangqi::SearchResult second = cached_searcher.search(second_position, cached_limits);
    const xiangqi::SearchResult uncached =
        uncached_searcher.search(uncached_position, uncached_limits);

    expect(first.tt_hits > 0, "iterative deepening produces transposition-table hits");
    expect(first.tt_move_orderings > 0,
           "a cached best move is reused for ordering when its score is too shallow");
    expect(second.tt_cutoffs > 0, "a repeated search reuses cached score bounds");
    expect(second.nodes < uncached.nodes,
           "cached repeated search visits fewer nodes than an uncached search");
    expect(second.score == uncached.score && second.best_move == uncached.best_move,
           "transposition table preserves score and best-move correctness");
    expect(first_position == second_position && second_position == uncached_position,
           "cached and uncached searches both restore the root position");
}

void test_search_respects_terminal_history() {
    const Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/9/4K4 w");
    xiangqi::GameHistory history(position, xiangqi::kNoCaptureDrawPlies);
    const std::uint64_t context_before = history.rule_context_hash();
    xiangqi::Searcher searcher;

    const xiangqi::SearchResult result = searcher.search(
        history,
        xiangqi::SearchLimits{.depth = 3,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});

    expect(!result.best_move.has_value() && result.score == 0,
           "an already adjudicated 60-move draw ends search at the root");
    expect(history.position() == position && history.ply_count() == 0 &&
               history.rule_context_hash() == context_before,
           "history-aware root adjudication leaves the supplied game unchanged");
}

void test_search_uses_perpetual_chase_result() {
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/4p4/r8/9/PP7/1C7/4K4 w"));
    const std::string prefix[]{
        "b1a1", "a4b4", "a1b1", "b4a4",
        "b1a1", "a4b4", "a1b1",
    };
    for (const std::string& text : prefix) {
        expect(history.play(move(text)), "search chase-prefix move is legal: " + text);
    }

    const Position before = history.position();
    const std::size_t plies_before = history.ply_count();
    const std::uint64_t context_before = history.rule_context_hash();
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult result = searcher.search(
        history,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});

    expect(result.best_move.has_value() && *result.best_move == move("b4a4"),
           "black completes the repetition that makes the red chaser lose");
    expect(result.score >= xiangqi::kMateThreshold,
           "winning a perpetual-chase adjudication receives a decisive score");
    expect(history.position() == before && history.ply_count() == plies_before &&
               history.rule_context_hash() == context_before,
           "perpetual-chase search restores board and history after exploring");
}

void test_search_avoids_own_perpetual_chase() {
    xiangqi::GameHistory history(Position::from_fen(
        "4k4/9/9/9/4p4/r8/9/PP7/1C7/4K4 w"));
    const std::string prefix[]{
        "b1a1", "a4b4", "a1b1", "b4a4", "b1a1", "a4b4",
    };
    for (const std::string& text : prefix) {
        expect(history.play(move(text)), "avoid-chase prefix move is legal: " + text);
    }

    const Position before = history.position();
    const std::size_t plies_before = history.ply_count();
    xiangqi::Searcher searcher;
    const xiangqi::SearchResult result = searcher.search(
        history,
        xiangqi::SearchLimits{.depth = 2,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});

    expect(result.best_move.has_value() && *result.best_move != move("a1b1"),
           "red avoids continuing a line in which its own long chase loses");
    expect(history.position() == before && history.ply_count() == plies_before,
           "avoiding a long chase leaves the caller's history unchanged");
}

void test_transposition_table_separates_rule_contexts() {
    const Position position = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/9/4K4 w");
    xiangqi::Searcher searcher;
    Position fresh = position;
    static_cast<void>(searcher.search(
        fresh,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta}));

    xiangqi::GameHistory near_limit(position, xiangqi::kNoCaptureDrawPlies - 1);
    const xiangqi::SearchResult result = searcher.search(
        near_limit,
        xiangqi::SearchLimits{.depth = 1,
                              .algorithm = xiangqi::SearchAlgorithm::AlphaBeta});
    expect(result.score == 0,
           "a cached board score cannot override a different 60-move context");
}

}  // namespace

int main() {
    test_initial_position();
    test_make_and_undo();
    test_horse_leg();
    test_cannon_screen();
    test_elephant_and_pawn_boundaries();
    test_flying_generals_and_pin();
    test_check_evasion();
    test_invalid_fen();
    test_halfka_features();
    test_game_history_and_repetition();
    test_history_check_and_capture_metadata();
    test_cycle_extraction_and_adjudication();
    test_perpetual_check_adjudication();
    test_perpetual_chase_adjudication();
    test_unprotected_piece_perpetual_chase();
    test_pawn_perpetual_chase_is_allowed();
    test_king_perpetual_chase_is_allowed();
    test_protected_piece_is_not_perpetually_chased();
    test_no_capture_draw_adjudication();
    test_evaluation_symmetry_and_material();
    test_evaluator_interface_and_search_injection();
    test_nnue_accumulator_and_forward_inference();
    test_nnue_incremental_accumulator_updates();
    test_search_maintains_evaluation_state();
    test_nnue_file_round_trip_and_search();
    test_advanced_pawn_value();
    test_piece_base_values();
    test_capture_move_ordering();
    test_static_exchange_evaluation();
    test_quiet_check_ordering();
    test_killer_counter_and_history_ordering();
    test_piece_to_history_ordering();
    test_search_finds_obvious_capture();
    test_negamax_matches_alpha_beta();
    test_search_terminal_and_depth_zero();
    test_quiescence_avoids_poisoned_capture();
    test_quiescence_searches_check_evasions();
    test_iterative_deepening_results();
    test_transposition_table_storage_and_replacement();
    test_search_uses_transposition_table();
    test_search_respects_terminal_history();
    test_search_uses_perpetual_chase_result();
    test_search_avoids_own_perpetual_chase();
    test_transposition_table_separates_rule_contexts();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All rule-engine tests passed.\n";
    return EXIT_SUCCESS;
}
