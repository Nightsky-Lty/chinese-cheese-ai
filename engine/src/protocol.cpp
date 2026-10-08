#include "xiangqi/cycle_adjudicator.hpp"
#include "xiangqi/evaluation.hpp"
#include "xiangqi/game_history.hpp"
#include "xiangqi/nnue.hpp"
#include "xiangqi/search.hpp"
#include "xiangqi/types.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

/// @brief 输出一行协议响应并立即刷新，避免 Python 客户端等待缓冲区。
/// @param response 不包含换行符的协议响应。
void respond(const std::string& response) {
    std::cout << response << '\n' << std::flush;
}

/// @brief 判断字符串是否以指定命令前缀开头。
/// @param text 完整输入行。
/// @param prefix 命令及其后空格组成的前缀。
bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

/// @brief 将当前对局状态编码成单行协议响应。
/// @param history 当前对局历史。
std::string state_response(const xiangqi::GameHistory& history) {
    std::ostringstream output;
    output << "state fen " << history.position().to_fen()
           << " ply " << history.ply_count()
           << " nocapture " << history.no_capture_plies();
    return output.str();
}

/// @brief 将当前对局裁决编码成稳定、便于 GUI 解析的单行响应。
/// @param history 当前对局历史。
/// @return ``result <outcome> reason <reason>`` 格式的响应。
std::string result_response(const xiangqi::GameHistory& history) {
    using xiangqi::HistoryVerdict;
    switch (xiangqi::adjudicate_history(history)) {
        case HistoryVerdict::DrawByThreefoldRepetition:
            return "result draw reason threefold_repetition";
        case HistoryVerdict::DrawByNoCapture:
            return "result draw reason no_capture_120_plies";
        case HistoryVerdict::RedWinsByBlackPerpetualCheck:
            return "result red_win reason black_perpetual_check";
        case HistoryVerdict::BlackWinsByRedPerpetualCheck:
            return "result black_win reason red_perpetual_check";
        case HistoryVerdict::RedWinsByBlackPerpetualChase:
            return "result red_win reason black_perpetual_chase";
        case HistoryVerdict::BlackWinsByRedPerpetualChase:
            return "result black_win reason red_perpetual_chase";
        case HistoryVerdict::Ongoing:
            break;
    }

    xiangqi::Position position = history.position();
    const xiangqi::GameStatus status = position.status();
    if (status == xiangqi::GameStatus::Ongoing) {
        return "result ongoing reason none";
    }
    const bool red_wins = position.side_to_move() == xiangqi::Color::Black;
    const std::string outcome = red_wins ? "red_win" : "black_win";
    const std::string reason = status == xiangqi::GameStatus::Checkmate
                                   ? "checkmate"
                                   : "no_legal_move";
    return "result " + outcome + " reason " + reason;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::optional<std::filesystem::path> nnue_path;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--nnue" && index + 1 < argc) {
                nnue_path = argv[++index];
            } else {
                throw std::invalid_argument("unknown or incomplete option: " + argument);
            }
        }

        std::optional<xiangqi::NnueEvaluator> nnue_evaluator;
        const xiangqi::Evaluator* evaluator = &xiangqi::handcrafted_evaluator();
        if (nnue_path) {
            nnue_evaluator.emplace(*nnue_path);
            evaluator = &*nnue_evaluator;
        }

        xiangqi::GameHistory history;
        xiangqi::Searcher searcher(*evaluator);
        respond("ready xiangqi-protocol 1");

        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                if (line == "quit") {
                    respond("bye");
                    return 0;
                }
                if (line == "ping") {
                    respond("pong");
                    continue;
                }
                if (line == "state") {
                    respond(state_response(history));
                    continue;
                }
                if (line == "result") {
                    respond(result_response(history));
                    continue;
                }
                if (line == "legal") {
                    xiangqi::Position position = history.position();
                    const auto moves = position.generate_legal_moves();
                    std::ostringstream output;
                    output << "legal " << moves.size();
                    for (xiangqi::Move move : moves) {
                        output << ' ' << xiangqi::move_to_string(move);
                    }
                    respond(output.str());
                    continue;
                }
                if (line == "undo") {
                    if (!history.undo_last()) {
                        respond("error no_history nothing to undo");
                    } else {
                        respond("ok undo " + state_response(history));
                    }
                    continue;
                }
                if (starts_with(line, "position fen ")) {
                    const std::string fen = line.substr(std::string("position fen ").size());
                    history.reset(xiangqi::Position::from_fen(fen));
                    searcher.clear_transposition_table();
                    respond("ok position " + state_response(history));
                    continue;
                }
                if (starts_with(line, "move ")) {
                    const std::string move_text = line.substr(std::string("move ").size());
                    const auto move = xiangqi::move_from_string(move_text);
                    if (!move) {
                        respond("error invalid_move move must look like b0c2");
                    } else if (!history.play(*move)) {
                        respond("error illegal_move move is not legal in current position");
                    } else {
                        respond("ok move " + move_text + " " + state_response(history));
                    }
                    continue;
                }
                if (starts_with(line, "go depth ")) {
                    const std::string depth_text = line.substr(std::string("go depth ").size());
                    std::size_t consumed = 0;
                    const int depth = std::stoi(depth_text, &consumed);
                    if (consumed != depth_text.size() || depth <= 0) {
                        respond("error invalid_depth depth must be a positive integer");
                        continue;
                    }
                    const xiangqi::SearchResult result = searcher.search(
                        history, xiangqi::SearchLimits{.depth = depth});
                    std::ostringstream output;
                    output << "bestmove ";
                    if (result.best_move) {
                        output << xiangqi::move_to_string(*result.best_move);
                    } else {
                        output << "none";
                    }
                    output << " score " << result.score
                           << " depth " << result.depth
                           << " nodes " << result.nodes;
                    respond(output.str());
                    continue;
                }
                respond("error unknown_command unsupported protocol command");
            } catch (const std::exception& error) {
                respond(std::string("error command_failed ") + error.what());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
