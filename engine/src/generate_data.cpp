#include "xiangqi/training_data.hpp"
#include "xiangqi/nnue.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

/// @brief 打印训练数据生成器的命令行用法。
void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  xiangqi_generate_data label --input FENS --output DATA [options]\n"
        << "  xiangqi_generate_data selfplay --output DATA [options]\n\n"
        << "Common options:\n"
        << "  --qdepth N             quiescence depth (default: 32)\n"
        << "  --max-score N          discard |score| above N (default: 28000)\n"
        << "  --weight X             sample weight (default: 1.0)\n"
        << "  --tt-mb N              transposition table MB per searcher (default: 16)\n"
        << "  --nnue MODEL           use an NNUE model instead of handcrafted evaluation\n"
        << "  --append               append instead of refusing an existing output\n"
        << "  --no-deduplicate       keep duplicate board positions\n\n"
        << "Label options:\n"
        << "  --depth N              labeling depth (default: 6)\n\n"
        << "Self-play options:\n"
        << "  --games N              number of games (default: 1)\n"
        << "  --play-depth N         move-selection depth (default: 4)\n"
        << "  --label-depth N        teacher-label depth (default: 6)\n"
        << "  --max-plies N          maximum plies per game (default: 240)\n"
        << "  --sample-start N       first sampled ply (default: 8)\n"
        << "  --sample-min-gap N     minimum sampling gap (default: 2)\n"
        << "  --sample-max-gap N     maximum sampling gap (default: 4)\n"
        << "  --random-plies N       diverse opening plies (default: 10)\n"
        << "  --random-top-k N       maximum near-best candidates (default: 3)\n"
        << "  --random-margin N      allowed score loss in opening (default: 150)\n"
        << "  --seed N               deterministic random seed (default: 1)\n";
}

/// @brief 将命令行整数解析为目标无符号类型。
template <typename Integer>
Integer parse_unsigned(std::string_view text, std::string_view option) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return value;
}

/// @brief 将命令行整数解析为 `int`。
int parse_int(std::string_view text, std::string_view option) {
    int value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return value;
}

/// @brief 将命令行浮点数解析为 `double`。
double parse_double(const std::string& text, std::string_view option) {
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    if (parsed != text.size()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return value;
}

/// @brief 读取必须紧跟在选项后的参数。
std::string require_value(int& index, int argc, char** argv,
                          std::string_view option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + std::string(option));
    }
    return argv[++index];
}

/// @brief 以安全模式打开输出文件，默认拒绝覆盖已有数据。
std::ofstream open_output(const std::filesystem::path& path, bool append) {
    if (!append && std::filesystem::exists(path)) {
        throw std::invalid_argument(
            "output already exists; use --append to add records: " + path.string());
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, append ? std::ios::app : std::ios::out);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help" ||
            std::string_view(argv[1]) == "-h") {
            print_usage(argc < 2 ? std::cerr : std::cout);
            return argc < 2 ? 1 : 0;
        }

        const std::string mode = argv[1];
        if (mode != "label" && mode != "selfplay") {
            throw std::invalid_argument("mode must be 'label' or 'selfplay'");
        }

        xiangqi::LabelPositionsConfig label_config;
        xiangqi::SelfPlayConfig selfplay_config;
        std::filesystem::path input_path;
        std::filesystem::path output_path;
        std::filesystem::path nnue_path;
        std::size_t tt_mb = 16;
        bool append = false;

        for (int index = 2; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--input") {
                input_path = require_value(index, argc, argv, option);
            } else if (option == "--output") {
                output_path = require_value(index, argc, argv, option);
            } else if (option == "--depth") {
                label_config.depth = parse_int(require_value(index, argc, argv, option), option);
            } else if (option == "--play-depth") {
                selfplay_config.play_depth = parse_int(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--label-depth") {
                selfplay_config.label_depth = parse_int(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--qdepth") {
                const int value = parse_int(require_value(index, argc, argv, option), option);
                label_config.quiescence_depth = value;
                selfplay_config.quiescence_depth = value;
            } else if (option == "--max-score") {
                const int value = parse_int(require_value(index, argc, argv, option), option);
                label_config.maximum_absolute_score = value;
                selfplay_config.maximum_absolute_score = value;
            } else if (option == "--weight") {
                const double value = parse_double(
                    require_value(index, argc, argv, option), option);
                label_config.weight = value;
                selfplay_config.weight = value;
            } else if (option == "--tt-mb") {
                tt_mb = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--nnue") {
                nnue_path = require_value(index, argc, argv, option);
            } else if (option == "--games") {
                selfplay_config.games = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--max-plies") {
                selfplay_config.maximum_plies = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--sample-start") {
                selfplay_config.sample_start_ply = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--sample-min-gap") {
                selfplay_config.minimum_sample_gap = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--sample-max-gap") {
                selfplay_config.maximum_sample_gap = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--random-plies") {
                selfplay_config.random_opening_plies = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--random-top-k") {
                selfplay_config.random_top_k = parse_unsigned<std::size_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--random-margin") {
                selfplay_config.random_score_margin = parse_int(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--seed") {
                selfplay_config.seed = parse_unsigned<std::uint64_t>(
                    require_value(index, argc, argv, option), option);
            } else if (option == "--append") {
                append = true;
            } else if (option == "--no-deduplicate") {
                label_config.deduplicate = false;
                selfplay_config.deduplicate = false;
            } else if (option == "--help" || option == "-h") {
                print_usage(std::cout);
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }

        if (output_path.empty()) {
            throw std::invalid_argument("--output is required");
        }
        std::optional<xiangqi::NnueEvaluator> nnue_evaluator;
        const xiangqi::Evaluator* evaluator = &xiangqi::handcrafted_evaluator();
        if (!nnue_path.empty()) {
            nnue_evaluator.emplace(nnue_path);
            evaluator = &*nnue_evaluator;
        }
        xiangqi::TrainingDataGenerator generator(*evaluator, tt_mb);
        std::ofstream output = open_output(output_path, append);
        xiangqi::TrainingDataStats stats;

        if (mode == "label") {
            if (input_path.empty()) {
                throw std::invalid_argument("label mode requires --input");
            }
            std::ifstream input(input_path);
            if (!input) {
                throw std::runtime_error("cannot open FEN input: " + input_path.string());
            }
            stats = generator.label_positions(input, output, label_config);
        } else {
            if (!input_path.empty()) {
                throw std::invalid_argument("selfplay mode does not accept --input");
            }
            stats = generator.self_play(output, selfplay_config);
        }

        std::cout << "games=" << stats.games
                  << " input_positions=" << stats.input_positions
                  << " samples=" << stats.samples_written
                  << " duplicates=" << stats.duplicate_positions
                  << " terminal=" << stats.terminal_positions
                  << " extreme_scores=" << stats.extreme_scores << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
