#include "xiangqi/evaluation.hpp"
#include "xiangqi/position.hpp"
#include "xiangqi/search.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        xiangqi::Position position = xiangqi::Position::initial();
        int search_depth = 0;
        xiangqi::SearchAlgorithm algorithm = xiangqi::SearchAlgorithm::AlphaBeta;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--fen" && index + 1 < argc) {
                position = xiangqi::Position::from_fen(argv[++index]);
            } else if (argument == "--depth" && index + 1 < argc) {
                search_depth = std::stoi(argv[++index]);
            } else if (argument == "--negamax") {
                algorithm = xiangqi::SearchAlgorithm::Negamax;
            } else {
                throw std::invalid_argument("unknown or incomplete command-line option: " + argument);
            }
        }

        std::cout << position.pretty();
        const xiangqi::EvaluationBreakdown evaluation =
            xiangqi::evaluate_breakdown(position);
        std::cout << "evaluation (side to move): " << xiangqi::evaluate(position)
                  << " [material(red-black): " << evaluation.material
                  << ", positional(red-black): " << evaluation.positional << "]\n";
        const auto moves = position.generate_legal_moves();
        std::cout << "legal moves: " << moves.size() << '\n';
        for (xiangqi::Move move : moves) {
            std::cout << xiangqi::move_to_string(move) << ' ';
        }
        std::cout << '\n';

        if (search_depth > 0) {
            xiangqi::Searcher searcher;
            const xiangqi::SearchResult result = searcher.search(
                position,
                xiangqi::SearchLimits{.depth = search_depth, .algorithm = algorithm});
            for (const xiangqi::IterationResult& iteration : result.iterations) {
                std::cout << "info depth " << iteration.depth
                          << " score " << iteration.score
                          << " nodes " << iteration.nodes
                          << " qnodes " << iteration.quiescence_nodes
                          << " cutoffs " << iteration.beta_cutoffs
                          << " tthits " << iteration.tt_hits
                          << " ttcutoffs " << iteration.tt_cutoffs
                          << " ttmoves " << iteration.tt_move_orderings
                          << " pvsresearches " << iteration.pvs_researches;
                if (iteration.best_move) {
                    std::cout << " bestmove "
                              << xiangqi::move_to_string(*iteration.best_move);
                }
                std::cout << '\n';
            }
            std::cout << "search: "
                      << (algorithm == xiangqi::SearchAlgorithm::AlphaBeta
                              ? "alpha-beta/pvs"
                              : "negamax")
                      << " depth " << result.depth
                      << " score " << result.score
                      << " nodes " << result.nodes
                      << " qnodes " << result.quiescence_nodes
                      << " cutoffs " << result.beta_cutoffs
                      << " tthits " << result.tt_hits
                      << " ttcutoffs " << result.tt_cutoffs
                      << " ttmoves " << result.tt_move_orderings
                      << " pvsresearches " << result.pvs_researches;
            if (result.best_move) {
                std::cout << " bestmove " << xiangqi::move_to_string(*result.best_move);
            }
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
