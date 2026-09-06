#include "xiangqi/cycle_adjudicator.hpp"
#include "xiangqi/evaluation.hpp"
#include "xiangqi/position.hpp"
#include "xiangqi/game_history.hpp"
#include "xiangqi/search.hpp"
#include "xiangqi/transposition_table.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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

void test_game_history_and_repetition() {
    const Position initial = Position::from_fen(
        "4k4/9/9/9/4p4/9/9/9/R8/4K4 w");
    xiangqi::GameHistory history(initial);

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
    expect(second.tt_cutoffs > 0, "a repeated search reuses cached score bounds");
    expect(second.nodes < uncached.nodes,
           "cached repeated search visits fewer nodes than an uncached search");
    expect(second.score == uncached.score && second.best_move == uncached.best_move,
           "transposition table preserves score and best-move correctness");
    expect(first_position == second_position && second_position == uncached_position,
           "cached and uncached searches both restore the root position");
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
    test_game_history_and_repetition();
    test_history_check_and_capture_metadata();
    test_cycle_extraction_and_adjudication();
    test_perpetual_check_adjudication();
    test_no_capture_draw_adjudication();
    test_evaluation_symmetry_and_material();
    test_advanced_pawn_value();
    test_piece_base_values();
    test_search_finds_obvious_capture();
    test_negamax_matches_alpha_beta();
    test_search_terminal_and_depth_zero();
    test_quiescence_avoids_poisoned_capture();
    test_quiescence_searches_check_evasions();
    test_iterative_deepening_results();
    test_transposition_table_storage_and_replacement();
    test_search_uses_transposition_table();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All rule-engine tests passed.\n";
    return EXIT_SUCCESS;
}
